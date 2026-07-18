#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {
namespace speculative {

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;
    output.all_probs = torch::softmax(logits, -1);

    std::tuple<torch::Tensor, torch::Tensor> sample_res;
    if (top_k == 1) {
        sample_res = torch::max(output.all_probs, -1, true);
    } else {
        sample_res = torch::topk(output.all_probs, top_k, -1);
    }

    output.token_ids = std::get<1>(sample_res);

    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) {
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output);

    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output) const {
    torch::Device target_device = getTorchCudaDevice();
    torch::Device host_device   = torch::Device(torch::kCPU);

    int batch_size = streams.size();

    // target_sampler_output.token_ids may be a CUDA tensor (Sampler keeps it on GPU to avoid
    // D2H sync during sampling). Move to CPU once here for data_ptr access.
    const torch::Tensor target_token_ids_cpu = target_sampler_output.token_ids.is_cuda() ?
                                                   target_sampler_output.token_ids.to(host_device, true) :
                                                   target_sampler_output.token_ids;
    const int*          new_all_token_ids    = target_token_ids_cpu.data_ptr<int32_t>();
    const size_t        token_stride         = target_token_ids_cpu.size(1);

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    // prepare data for chain speculative sampling
    auto          draft_token_ids_d_t    = draft_token_ids.to(target_device).clone();
    auto          draft_token_probs_d_t  = draft_token_probs;
    auto          target_token_probs_d_t = target_token_probs;
    auto          rand_options           = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d      = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

    // Override per-stream uniform samples with seeded generator when random_seed is set,
    // ensuring deterministic acceptance for reproducible iter_count.
    {
        int idx = 0;
        for (const auto& stream : streams) {
            auto gen = stream->getGenerator();
            if (gen.defined()) {
                uniform_samples_d[idx] = torch::rand({(long)propose_step_ + 1}, gen, std::nullopt, rand_options);
            }
            idx++;
        }
    }
    torch::Tensor output_token_ids_d     = torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                                                    torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    torch::Tensor output_accepted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));

    // Per-stream do_sample flag. Greedy streams (do_sample=false) must take the
    // deterministic accept path (accept a draft iff it equals the target argmax,
    // otherwise emit the target token) so that MTP greedy output is byte-identical
    // to non-speculative greedy. Sampling streams (do_sample=true) keep the random
    // rejection test. execChainSpeculativeSampling had no such branch — it always
    // ran the stochastic u*p<q test, making greedy non-equivalent (repetition
    // degeneration). execRejectionSampling implements the greedy special-case.
    // Greedy criterion matches vLLM's rejection sampler: a stream is greedy when
    // temperature == 0 OR top_k == 1 OR do_sample is explicitly off. RTP's
    // do_sample field defaults to true and is NOT lowered for temperature=0
    // requests, so keying on do_sample alone would leave temp=0 greedy requests
    // on the stochastic path. Treat any of the three greedy signals as do_sample=false.
    torch::Tensor do_sample_h = torch::empty({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool));
    {
        bool* ds  = do_sample_h.data_ptr<bool>();
        int   idx = 0;
        for (const auto& stream : streams) {
            const auto& cfg    = stream->generateConfig();
            const bool  greedy = !cfg->do_sample || cfg->temperature == 0.0f || cfg->top_k == 1;
            ds[idx]            = !greedy;
            idx++;
        }
    }
    torch::Tensor do_sample_d        = do_sample_h.to(target_device, true);
    torch::Tensor target_token_ids_d = target_token_ids.to(target_device).to(torch::kInt32).contiguous();

    execRejectionSampling({draft_token_probs_d_t,
                           draft_token_ids_d_t,
                           uniform_samples_d,
                           target_token_probs_d_t,
                           target_token_ids_d,
                           output_token_ids_d,
                           output_accepted_token_num_d,
                           do_sample_d});

    // back to host
    torch::Tensor output_token_ids_h          = output_token_ids_d.to(host_device, true);
    torch::Tensor output_accepted_token_num_h = output_accepted_token_num_d.to(host_device);  // implicit sync here

    torch::Tensor draft_token_ids_h;
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->forceSpAccept()) {
            draft_token_ids_h = draft_token_ids.cpu().clone();
            break;
        }
    }

    int stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        torch::Tensor accept_tokens;
        size_t        accept_len = 0;

        if (stream->forceSpAccept()) {
            accept_len    = propose_step_ + 1;
            accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
            memcpy(accept_tokens.data_ptr<int>(),
                   draft_token_ids_h.data_ptr<int32_t>() + stream_idx * propose_step_,
                   sizeof(int32_t) * propose_step_);
        } else {
            accept_len    = output_accepted_token_num_h[stream_idx].item<int32_t>();
            accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
            memcpy(accept_tokens.data_ptr<int>(),
                   output_token_ids_h[stream_idx].data_ptr<int32_t>(),
                   sizeof(int32_t) * accept_len);
        }

        // always use target token as the last token
        accept_tokens.data_ptr<int>()[accept_len - 1] =
            new_all_token_ids[(stream_idx * (propose_step_ + 1) + accept_len - 1) * token_stride + token_stride - 1];

        sample_output.accept_tokens.push_back(std::move(accept_tokens));
        sample_output.accept_len.push_back(accept_len);
        stream_idx++;
    }
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}
}  // namespace speculative
}  // namespace rtp_llm