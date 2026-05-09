package sample

import data.MetaInfo
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import turbo.TurboSampler
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

/**
 * 벡터 백엔드 체크포인트로 인터랙티브 대화 (ChatGPT 스타일).
 *
 * 사용:
 *   ```
 *   ./gradlew runChatVec --args="<ckpt-dir> [temp] [topK] [topP] [repPenalty]"
 *   ```
 *   - temp 0.8(기본). 0.5-0.7 권장 (topic relevance ↑)
 *   - topK 40(기본). 0이면 비활성
 *   - topP 1.0(기본, 비활성). 0.9-0.95 권장 (top-k 위에 nucleus cutoff)
 *   - repPenalty 1.0(기본, 비활성). 1.1-1.2 권장 (반복 차단)
 *
 * 동작:
 *   - 체크포인트 로드 후 stdin에서 한 줄씩 사용자 입력 받음
 *   - 사용자 turn을 conversation context에 누적: `"<user input>"<|turn|>`
 *   - 모델이 `<|turn|>` 또는 `<|eos|>`까지 생성 → 응답 출력
 *   - 모델 turn도 context에 누적
 *   - context가 model.blockSize 초과하면 앞쪽 자름 (BOS는 유지)
 *
 * 명령어:
 *   `/quit`        대화 종료
 *   `/reset`       대화 history 초기화 (`<|bos|>`로만 다시 시작)
 *   `/temp <n>`    temperature 조정 (예: /temp 0.5)
 *   `/topk <n>`    top-k 조정
 *   `/help`        명령어 안내
 *
 * 데이터 포맷이 발화 사이 `<|turn|>` 토큰을 가진 모델(예: conv-mix-turn)에서만
 * single-turn stop이 의미 있음. `<|turn|>` 없는 ckpt면 EOS까지 generate.
 */
fun main(args: Array<String>) {
    require(args.isNotEmpty()) { "사용: <ckpt-dir> [temp=0.8] [topK=40] [topP=1.0] [repPenalty=1.0]" }
    val ckptDir = File(args[0])
    val temp = args.getOrNull(1)?.toFloatOrNull() ?: 0.8f
    val topK = args.getOrNull(2)?.toIntOrNull() ?: 40
    val topP = args.getOrNull(3)?.toFloatOrNull() ?: 1.0f
    val repPenalty = args.getOrNull(4)?.toFloatOrNull() ?: 1.0f
    require(ckptDir.exists()) { "ckpt 경로 없음: ${ckptDir.absolutePath}" }

    // meta.json에서 special token id 추출
    val metaFile = File(ckptDir, "meta.json")
    require(metaFile.exists()) { "meta.json 없음" }
    val parser = Json { ignoreUnknownKeys = true }
    val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
    val eosId = meta.stringToIndex["<|eos|>"] ?: 0
    val bosId = meta.stringToIndex["<|bos|>"] ?: -1
    val turnId = meta.stringToIndex["<|turn|>"]

    if (turnId == null) {
        println("⚠️  meta.json에 <|turn|> 토큰 없음 — single-turn stop 비활성. EOS까지 생성.")
    } else {
        println("✓ <|turn|> id = $turnId, EOS id = $eosId — single-turn 응답 모드")
    }

    val stopIds = mutableListOf(eosId).also { if (turnId != null) it.add(turnId) }

    val config = SampleConfig(
        modelDirectoryPath = ckptDir.absolutePath,
        numberOfSamples = 1,
        maximumNewTokens = 200,
        samplingTemperature = temp,
        topKFilteringSize = topK,
        topProbabilityThreshold = topP,
        repetitionPenalty = repPenalty,
        stopTokenIds = stopIds,
    )
    val sampler = TurboSampler(config)
    val maxCtx = sampler.maxContextLength
    println("# blockSize = $maxCtx, temp=$temp, topK=$topK, topP=$topP, repPenalty=$repPenalty")
    println("# /help 로 명령어 보기. /quit 으로 종료.")
    println()

    // 대화 history 누적용 토큰 id 리스트
    val history = mutableListOf<Int>()
    if (bosId >= 0) history += bosId

    val br = BufferedReader(InputStreamReader(System.`in`))
    var currentTemp = temp
    var currentTopK = topK

    while (true) {
        print("You > ")
        System.out.flush()
        val raw = br.readLine() ?: break
        val line = raw.trim()
        if (line.isEmpty()) continue

        when {
            line == "/quit" || line == "/exit" -> {
                println("# 종료")
                return
            }
            line == "/help" -> {
                println("""
                |명령어:
                |  /quit, /exit   대화 종료
                |  /reset         history 초기화
                |  /temp <n>      temperature (현재: $currentTemp)
                |  /topk <n>      top-k (현재: $currentTopK)
                |  /help          이 도움말
                """.trimMargin())
                continue
            }
            line == "/reset" -> {
                history.clear()
                if (bosId >= 0) history += bosId
                println("# history 초기화됨")
                continue
            }
            line.startsWith("/temp ") -> {
                println("# temp/topK는 시작 시 인자로만 설정 가능 (모델 재로드 비용 회피). " +
                    "현재: temp=$currentTemp, topK=$currentTopK")
                continue
            }
            line.startsWith("/topk ") -> {
                println("# temp/topK는 시작 시 인자로만 설정 가능. 현재: temp=$currentTemp, topK=$currentTopK")
                continue
            }
        }

        // 사용자 turn을 history에 추가: <user text><|turn|>
        // 학습 데이터(conv-mix-turn-noq 계열)가 따옴표 제거된 형식이라 quote wrap 안 함.
        // 따옴표 wrap 했던 이전 버전은 distribution shift로 모델 응답이 망가졌음.
        val userTurnIds = sampler.encodeText(line).toMutableList()
        if (turnId != null) userTurnIds += turnId
        history += userTurnIds

        // blockSize 초과하면 앞쪽 (BOS 직후) 자름. blockSize의 70%까지만 prompt에 사용해
        // 응답 생성 여유 확보.
        val maxPrompt = (maxCtx * 0.7).toInt().coerceAtLeast(8)
        var promptArr = if (history.size > maxPrompt) {
            // BOS 보존하며 뒤쪽 maxPrompt-1 유지
            val tail = history.takeLast(maxPrompt - 1)
            val withBos = if (bosId >= 0) (listOf(bosId) + tail) else tail
            withBos.toIntArray()
        } else {
            history.toIntArray()
        }

        val (newIds, response) = sampler.continueOne(promptArr)

        println("Bot > $response")
        println()

        // 모델 turn을 history에 추가 (turn 토큰까지)
        history += newIds
        if (turnId != null) history += turnId
    }
}
