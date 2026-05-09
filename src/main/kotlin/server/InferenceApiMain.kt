package server

import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import sample.SampleConfig
import turbo.TurboSampler

@Serializable
data class GenerateRequest(
    val prompt: String,
    val maxNewTokens: Int? = null,
    val temperature: Float? = null,
    val topK: Int? = null,
    val topP: Float? = null,
    val repetitionPenalty: Float? = null,
    val numberOfSamples: Int? = null,
    val stopTokenIds: List<Int>? = null,
)

@Serializable
data class GenerateResponse(val results: List<String>)

@Serializable
data class HealthResponse(
    val status: String,
    val modelDir: String,
    val maxContext: Int,
)

@Serializable
data class ErrorResponse(val error: String)

fun main(args: Array<String>) {
    require(args.isNotEmpty()) {
        "사용법: runInferenceApi --args=\"<체크포인트 디렉토리> [포트=8080]\""
    }
    val ckptDir = args[0]
    val port = args.getOrNull(1)?.toIntOrNull() ?: 8080

    val baseConfig = SampleConfig(
        modelDirectoryPath = ckptDir,
        numberOfSamples = 1,
        maximumNewTokens = 100,
        samplingTemperature = 0.8f,
        topKFilteringSize = 40,
    )
    println("# Loading Turbo sampler from $ckptDir")
    val sampler = TurboSampler(baseConfig)
    val mutex = Mutex()

    println("# Listening on http://0.0.0.0:$port (POST /generate, GET /health)")
    embeddedServer(Netty, port = port, host = "0.0.0.0") {
        install(ContentNegotiation) { json() }
        routing {
            get("/health") {
                call.respond(
                    HealthResponse(
                        status = "ok",
                        modelDir = ckptDir,
                        maxContext = sampler.maxContextLength,
                    )
                )
            }
            post("/generate") {
                val req = call.receive<GenerateRequest>()
                if (req.prompt.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, ErrorResponse("prompt가 비어있음"))
                    return@post
                }
                val cfg = baseConfig.copy(
                    maximumNewTokens = req.maxNewTokens ?: baseConfig.maximumNewTokens,
                    samplingTemperature = req.temperature ?: baseConfig.samplingTemperature,
                    topKFilteringSize = req.topK ?: baseConfig.topKFilteringSize,
                    topProbabilityThreshold = req.topP ?: baseConfig.topProbabilityThreshold,
                    repetitionPenalty = req.repetitionPenalty ?: baseConfig.repetitionPenalty,
                    numberOfSamples = req.numberOfSamples ?: baseConfig.numberOfSamples,
                    stopTokenIds = req.stopTokenIds ?: baseConfig.stopTokenIds,
                )
                val results = mutex.withLock { sampler.generate(req.prompt, cfg) }
                call.respond(GenerateResponse(results))
            }
        }
    }.start(wait = true)
}
