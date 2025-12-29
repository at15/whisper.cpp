// Stream server for real-time audio transcription with VAD support
// Uses HTTP polling instead of WebSocket for simplicity

#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <csignal>
#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

// Log macro for debugging
#define LOG(fmt, ...) fprintf(stderr, "[streamserver] " fmt "\n", ##__VA_ARGS__)

namespace {

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// Server configuration
struct server_params {
    std::string hostname = "127.0.0.1";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
    int32_t session_timeout_s = 60;
};

// Whisper configuration (command-line defaults)
struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    // Streaming parameters
    int32_t step_ms = 500;      // Process every N ms of new audio
    int32_t keep_ms = 200;      // Overlap from previous chunk
    int32_t length_ms = 10000;  // Max audio chunk for inference

    // VAD parameters
    float vad_thold = 0.01f;    // Energy threshold for VAD
    float freq_thold = 100.0f;  // High-pass filter cutoff Hz

    bool use_gpu = true;
    bool flash_attn = true;

    std::string model = "models/ggml-base.en.bin";
};

// Transcript segment
struct TranscriptSegment {
    int index;
    std::string text;
    int64_t t0;  // Start time in centiseconds
    int64_t t1;  // End time in centiseconds
};

// Stream session state
struct StreamSession {
    std::string session_id;

    // Audio buffers
    std::vector<float> pcmf32;      // Current processing buffer
    std::vector<float> pcmf32_old;  // Previous buffer (for overlap)
    std::vector<float> pcmf32_new;  // Newly received audio

    int last_chunk_id = -1;
    std::chrono::steady_clock::time_point last_activity;

    // Configuration (set at creation via /stream/create)
    std::string language = "auto";
    bool translate = false;

    // Transcription state
    std::vector<TranscriptSegment> segments;
    std::string current_text;
    int64_t current_t0 = 0;
    int stream_ms = 0;
    bool active_speech = false;
    int n_iter = 0;
};

// Global state
std::unordered_map<std::string, StreamSession> sessions;
std::mutex sessions_mutex;
std::mutex whisper_mutex;

// Calculate number of samples from milliseconds
inline int ms_to_samples(int ms, int sample_rate = WHISPER_SAMPLE_RATE) {
    return (sample_rate * ms) / 1000;
}

// Simple energy-based VAD
bool is_speech(const std::vector<float>& pcmf32, float energy_thold = 0.01f, float freq_thold = 100.0f) {
    if (pcmf32.empty()) {
        return false;
    }

    std::vector<float> filtered = pcmf32;

    // Apply high-pass filter to remove low-frequency noise
    if (freq_thold > 0.0f) {
        high_pass_filter(filtered, freq_thold, WHISPER_SAMPLE_RATE);
    }

    // Calculate RMS energy
    float energy = 0.0f;
    for (const auto& sample : filtered) {
        energy += sample * sample;
    }
    energy = std::sqrt(energy / filtered.size());

    return energy > energy_thold;
}

void print_usage(int argc, char** argv, const whisper_params& params, const server_params& sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME    [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  --step N                   [%-7d] process every N ms of new audio\n", params.step_ms);
    fprintf(stderr, "  --keep N                   [%-7d] overlap from previous chunk in ms\n", params.keep_ms);
    fprintf(stderr, "  --length N                 [%-7d] max audio chunk for inference in ms\n", params.length_ms);
    fprintf(stderr, "  --vad-thold N              [%-7.3f] VAD energy threshold\n", params.vad_thold);
    fprintf(stderr, "  --freq-thold N             [%-7.1f] high-pass filter cutoff Hz\n", params.freq_thold);
    fprintf(stderr, "  --host HOST                [%-7s] hostname\n", sparams.hostname.c_str());
    fprintf(stderr, "  --port PORT                [%-7d] port\n", sparams.port);
    fprintf(stderr, "  --timeout N                [%-7d] session timeout in seconds\n", sparams.session_timeout_s);
    fprintf(stderr, "  -ng,      --no-gpu         disable GPU\n");
    fprintf(stderr, "\n");
}

bool parse_params(int argc, char** argv, whisper_params& params, server_params& sparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params, sparams);
            exit(0);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "--step") {
            params.step_ms = std::stoi(argv[++i]);
        } else if (arg == "--keep") {
            params.keep_ms = std::stoi(argv[++i]);
        } else if (arg == "--length") {
            params.length_ms = std::stoi(argv[++i]);
        } else if (arg == "--vad-thold") {
            params.vad_thold = std::stof(argv[++i]);
        } else if (arg == "--freq-thold") {
            params.freq_thold = std::stof(argv[++i]);
        } else if (arg == "--host") {
            sparams.hostname = argv[++i];
        } else if (arg == "--port") {
            sparams.port = std::stoi(argv[++i]);
        } else if (arg == "--timeout") {
            sparams.session_timeout_s = std::stoi(argv[++i]);
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params, sparams);
            return false;
        }
    }
    return true;
}

// Process audio for a session and run inference if needed
void process_session_audio(StreamSession& session, whisper_context* ctx, const whisper_params& params) {
    const int n_samples_step = ms_to_samples(params.step_ms);
    const int n_samples_len = ms_to_samples(params.length_ms);
    const int n_samples_keep = ms_to_samples(params.keep_ms);
    const int n_new_line = std::max(1, params.length_ms / params.step_ms - 1);

    // Check if we have enough new audio to process
    if ((int)session.pcmf32_new.size() < n_samples_step) {
        return;
    }

    int n_samples_new = session.pcmf32_new.size();

    // Take up to length_ms audio from previous iteration
    int n_samples_take = std::min(
        (int)session.pcmf32_old.size(),
        std::max(0, n_samples_keep + n_samples_len - n_samples_new)
    );

    // Build processing buffer: [old_overlap | new_audio]
    session.pcmf32.resize(n_samples_take + n_samples_new);
    if (n_samples_take > 0) {
        std::copy(
            session.pcmf32_old.end() - n_samples_take,
            session.pcmf32_old.end(),
            session.pcmf32.begin()
        );
    }
    std::copy(
        session.pcmf32_new.begin(),
        session.pcmf32_new.end(),
        session.pcmf32.begin() + n_samples_take
    );

    session.pcmf32_old = session.pcmf32;

    // VAD check
    if (session.active_speech || is_speech(session.pcmf32, params.vad_thold, params.freq_thold)) {
        LOG("Session %s: speech detected, running inference (iter=%d, samples=%zu)",
            session.session_id.c_str(), session.n_iter, session.pcmf32.size());

        // Run inference
        std::lock_guard<std::mutex> lock(whisper_mutex);

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime = false;
        wparams.print_progress = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;
        wparams.translate = session.translate;
        wparams.language = session.language.c_str();
        wparams.n_threads = params.n_threads;
        wparams.single_segment = true;
        wparams.no_context = true;

        if (whisper_full(ctx, wparams, session.pcmf32.data(), session.pcmf32.size()) == 0) {
            // Extract results
            std::string text;
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                text += whisper_full_get_segment_text(ctx, i);
            }

            session.current_text = text;
            session.current_t0 = (session.stream_ms - (int)(session.pcmf32.size() * 1000 / WHISPER_SAMPLE_RATE)) / 10;

            LOG("Session %s: transcribed: %s", session.session_id.c_str(), text.c_str());
        } else {
            LOG("Session %s: inference failed", session.session_id.c_str());
        }

        session.n_iter++;
        session.active_speech = true;

        // Flush segment periodically
        if (session.n_iter % n_new_line == 0) {
            if (!session.current_text.empty()) {
                TranscriptSegment seg;
                seg.index = session.segments.size();
                seg.text = session.current_text;
                seg.t0 = session.current_t0;
                seg.t1 = session.stream_ms / 10;
                session.segments.push_back(seg);

                LOG("Session %s: segment %d finalized: %s",
                    session.session_id.c_str(), seg.index, seg.text.c_str());
            }

            // Keep tail of audio for next iteration
            session.pcmf32_old.assign(
                session.pcmf32.end() - std::min((int)session.pcmf32.size(), n_samples_keep),
                session.pcmf32.end()
            );

            session.current_text.clear();
            session.active_speech = false;
        }
    }

    // Clear new audio buffer
    session.pcmf32_new.clear();
}

// Cleanup expired sessions
void cleanup_expired_sessions(int timeout_s, std::atomic<bool>& running) {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(10));

        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(sessions_mutex);

        for (auto it = sessions.begin(); it != sessions.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - it->second.last_activity
            ).count();

            if (elapsed > timeout_s) {
                LOG("Session %s expired after %lld seconds", it->first.c_str(), (long long)elapsed);
                it = sessions.erase(it);
            } else {
                ++it;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    whisper_params params;
    server_params sparams;

    if (!parse_params(argc, argv, params, sparams)) {
        return 1;
    }

    // Validate parameters
    params.keep_ms = std::min(params.keep_ms, params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    LOG("Loading model: %s", params.model.c_str());

    // Initialize whisper context
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        LOG("Failed to initialize whisper context");
        return 1;
    }

    LOG("Model loaded successfully");

    // Create HTTP server
    auto svr = std::make_unique<httplib::Server>();

    svr->set_default_headers({
        {"Server", "whisper-streamserver"},
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "content-type, authorization"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}
    });

    // Start cleanup thread
    std::atomic<bool> cleanup_running{true};
    std::thread cleanup_thread(cleanup_expired_sessions, sparams.session_timeout_s, std::ref(cleanup_running));

    // OPTIONS handler for CORS
    svr->Options("/stream/.*", [](const Request&, Response&) {});

    // POST /stream/create - Create a new session
    svr->Post("/stream/create", [&](const Request& req, Response& res) {
        std::string session_id;
        std::string language = "auto";
        bool translate = false;

        // Parse JSON body or form data
        if (req.has_header("Content-Type") &&
            req.get_header_value("Content-Type").find("application/json") != std::string::npos) {
            try {
                auto body = json::parse(req.body);
                if (body.contains("session_id")) session_id = body["session_id"];
                if (body.contains("language")) language = body["language"];
                if (body.contains("translate")) translate = body["translate"];
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"status":"error","message":"invalid JSON"})", "application/json");
                return;
            }
        } else {
            if (req.has_param("session_id")) session_id = req.get_param_value("session_id");
            if (req.has_param("language")) language = req.get_param_value("language");
            if (req.has_param("translate")) translate = req.get_param_value("translate") == "true";
        }

        if (session_id.empty()) {
            res.status = 400;
            res.set_content(R"({"status":"error","message":"session_id is required"})", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(sessions_mutex);

        if (sessions.find(session_id) != sessions.end()) {
            LOG("Session %s already exists", session_id.c_str());
            res.status = 409;
            res.set_content(R"({"status":"error","message":"session already exists"})", "application/json");
            return;
        }

        StreamSession session;
        session.session_id = session_id;
        session.language = language;
        session.translate = translate;
        session.last_activity = std::chrono::steady_clock::now();

        sessions[session_id] = std::move(session);

        LOG("Session %s created (language=%s, translate=%s)",
            session_id.c_str(), language.c_str(), translate ? "true" : "false");

        json response = {
            {"status", "created"},
            {"session_id", session_id},
            {"language", language},
            {"translate", translate}
        };
        res.status = 201;
        res.set_content(response.dump(), "application/json");
    });

    // POST /stream/push - Upload audio chunk
    svr->Post("/stream/push", [&](const Request& req, Response& res) {
        std::string session_id;
        int chunk_id = -1;

        if (req.has_param("session_id")) session_id = req.get_param_value("session_id");
        if (req.has_param("chunk_id")) chunk_id = std::stoi(req.get_param_value("chunk_id"));

        if (session_id.empty()) {
            res.status = 400;
            res.set_content(R"({"status":"error","message":"session_id is required"})", "application/json");
            return;
        }

        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content(R"({"status":"error","message":"file is required"})", "application/json");
            return;
        }

        auto audio_file = req.get_file_value("file");

        // Find session
        std::lock_guard<std::mutex> lock(sessions_mutex);

        auto it = sessions.find(session_id);
        if (it == sessions.end()) {
            LOG("Session %s not found, push rejected (chunk_id=%d)", session_id.c_str(), chunk_id);
            res.status = 404;
            res.set_content(R"({"status":"error","message":"session not found or expired"})", "application/json");
            return;
        }

        StreamSession& session = it->second;

        // Check chunk ordering
        if (chunk_id >= 0 && chunk_id <= session.last_chunk_id) {
            LOG("Session %s: duplicate or out-of-order chunk %d (last=%d)",
                session_id.c_str(), chunk_id, session.last_chunk_id);
        }

        // Parse WAV audio
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        if (!::read_audio_data(audio_file.content, pcmf32, pcmf32s, false)) {
            LOG("Session %s: failed to parse audio data", session_id.c_str());
            res.status = 400;
            res.set_content(R"({"status":"error","message":"failed to parse audio data"})", "application/json");
            return;
        }

        // Append to session buffer
        session.pcmf32_new.insert(session.pcmf32_new.end(), pcmf32.begin(), pcmf32.end());
        session.stream_ms += (int)(pcmf32.size() * 1000 / WHISPER_SAMPLE_RATE);
        session.last_chunk_id = chunk_id;
        session.last_activity = std::chrono::steady_clock::now();

        LOG("Session %s: received chunk %d (%zu samples, total=%dms)",
            session_id.c_str(), chunk_id, pcmf32.size(), session.stream_ms);

        // Process audio if we have enough
        process_session_audio(session, ctx, params);

        json response = {
            {"status", "ok"},
            {"session_id", session_id},
            {"chunk_id", chunk_id},
            {"buffered_ms", session.stream_ms}
        };
        res.set_content(response.dump(), "application/json");
    });

    // GET /stream/poll - Retrieve transcription results
    svr->Get("/stream/poll", [&](const Request& req, Response& res) {
        std::string session_id;
        int after_index = -1;

        if (req.has_param("session_id")) session_id = req.get_param_value("session_id");
        if (req.has_param("after_index")) after_index = std::stoi(req.get_param_value("after_index"));

        if (session_id.empty()) {
            res.status = 400;
            res.set_content(R"({"status":"error","message":"session_id is required"})", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(sessions_mutex);

        auto it = sessions.find(session_id);
        if (it == sessions.end()) {
            res.status = 404;
            res.set_content(R"({"status":"error","message":"session not found or expired"})", "application/json");
            return;
        }

        const StreamSession& session = it->second;

        json response = {
            {"session_id", session_id},
            {"stream_ms", session.stream_ms},
            {"segments", json::array()},
            {"is_speaking", session.active_speech}
        };

        // Add segments after the specified index
        for (const auto& seg : session.segments) {
            if (seg.index > after_index) {
                response["segments"].push_back({
                    {"index", seg.index},
                    {"text", seg.text},
                    {"t0", seg.t0},
                    {"t1", seg.t1}
                });
            }
        }

        // Add current partial transcription
        if (!session.current_text.empty()) {
            response["current"] = {
                {"text", session.current_text},
                {"t0", session.current_t0}
            };
        }

        res.set_content(response.dump(), "application/json");
    });

    // DELETE /stream/:session_id - Close session
    svr->Delete(R"(/stream/([^/]+))", [&](const Request& req, Response& res) {
        std::string session_id = req.matches[1];

        std::lock_guard<std::mutex> lock(sessions_mutex);

        auto it = sessions.find(session_id);
        if (it == sessions.end()) {
            res.status = 404;
            res.set_content(R"({"status":"error","message":"session not found or expired"})", "application/json");
            return;
        }

        const StreamSession& session = it->second;

        // Build final response
        json response = {
            {"session_id", session_id},
            {"stream_ms", session.stream_ms},
            {"segments", json::array()},
            {"is_final", true}
        };

        for (const auto& seg : session.segments) {
            response["segments"].push_back({
                {"index", seg.index},
                {"text", seg.text},
                {"t0", seg.t0},
                {"t1", seg.t1}
            });
        }

        // Include any remaining current text as final segment
        if (!session.current_text.empty()) {
            response["segments"].push_back({
                {"index", (int)session.segments.size()},
                {"text", session.current_text},
                {"t0", session.current_t0},
                {"t1", session.stream_ms / 10}
            });
        }

        LOG("Session %s closed, %zu segments", session_id.c_str(), response["segments"].size());

        sessions.erase(it);

        res.set_content(response.dump(), "application/json");
    });

    // GET /stream/health - Health check
    svr->Get("/stream/health", [&](const Request&, Response& res) {
        std::lock_guard<std::mutex> lock(sessions_mutex);

        json response = {
            {"status", "ready"},
            {"active_sessions", sessions.size()}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Setup signal handler
    shutdown_handler = [&](int) {
        LOG("Shutting down...");
        svr->stop();
    };

#if defined(_WIN32)
    signal(SIGINT, signal_handler);
#else
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif

    // Start server
    LOG("Starting server on %s:%d", sparams.hostname.c_str(), sparams.port);
    LOG("Parameters: step=%dms, keep=%dms, length=%dms, vad_thold=%.3f",
        params.step_ms, params.keep_ms, params.length_ms, params.vad_thold);

    svr->set_read_timeout(sparams.read_timeout);
    svr->set_write_timeout(sparams.write_timeout);

    if (!svr->bind_to_port(sparams.hostname, sparams.port)) {
        LOG("Failed to bind to %s:%d", sparams.hostname.c_str(), sparams.port);
        cleanup_running = false;
        cleanup_thread.join();
        whisper_free(ctx);
        return 1;
    }

    std::thread server_thread([&]() {
        svr->listen_after_bind();
    });

    svr->wait_until_ready();
    LOG("Server is ready");

    server_thread.join();

    // Cleanup
    cleanup_running = false;
    cleanup_thread.join();
    whisper_free(ctx);

    LOG("Server stopped");
    return 0;
}
