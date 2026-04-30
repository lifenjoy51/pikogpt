package data

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `StoriesBPEPrep.run()`의 두 입력 경로를 검증.
 *
 * - **분리**: `train.txt` + `val.txt`가 있으면 각각 따로 encode되고 BPE는 train에서만 학습.
 * - **fallback**: `stories.txt`만 있으면 90:10 순차 절단.
 */
class StoriesBpePrepSplitTest {
    private val root = File("build/tmp/prep-split-test")

    @AfterTest
    fun cleanup() {
        root.deleteRecursively()
    }

    @Test
    fun separateInputsUseTrainForVocabAndBothForBins() {
        val dir = File(root, "separate").apply { mkdirs() }
        // train/val를 다른 문자 집합으로 구성 — val에만 있는 글자가 vocab에 없어야 leakage 아님.
        val trainText = "aaa bbb ccc ".repeat(200)
        val valText = "zzz yyy ".repeat(100)
        File(dir, "train.txt").writeText(trainText)
        File(dir, "val.txt").writeText(valText)

        StoriesBPEPrep.run(dir.absolutePath, verbose = false)

        val meta = Json { ignoreUnknownKeys = true }
            .decodeFromString<MetaInfo>(File(dir, "meta.json").readText())
        val chars = meta.stringToIndex.keys.flatMap { it.toList() }.toSet()
        // train에만 a/b/c, val에만 z/y — BPE 학습이 train에서만 됐으면 z/y는 단일 문자로도 등장 안 함.
        assertTrue('a' in chars || 'b' in chars || 'c' in chars, "train 글자 중 하나는 vocab에 있어야")
        assertTrue('z' !in chars && 'y' !in chars, "val-only 글자는 vocab에 없어야 (leakage 없음)")

        val trainTokens = readTokens(File(dir, "train.bin"))
        val valTokens = readTokens(File(dir, "val.bin"))
        assertTrue(trainTokens > 0 && valTokens > 0, "bin 파일 모두 생성되어야")

        // 분리 경로 확인: val이 90:10 cut이었다면 valTokens ≈ 0.1 × (train+val).
        // 실제로는 train.txt(abc반복) vs val.txt(zy반복, 전부 unk)로 독립 encode된 값이 나와야 함.
        // 구체적으로 90:10 cut 가정 시 valTokens ≈ 0.111 × trainTokens — 분리 경로면 그 값과 다름.
        val ratio = valTokens.toDouble() / (trainTokens + valTokens)
        assertTrue(ratio !in 0.095..0.11, "비율이 공교롭게 90:10과 같으면 fallback일 가능성 의심: $ratio")
    }

    @Test
    fun fallbackStoriesTxtSplits90to10() {
        val dir = File(root, "fallback").apply { mkdirs() }
        File(dir, "stories.txt").writeText("abc ".repeat(1000))

        StoriesBPEPrep.run(dir.absolutePath, verbose = false)

        val trainTokens = readTokens(File(dir, "train.bin"))
        val valTokens = readTokens(File(dir, "val.bin"))
        val total = trainTokens + valTokens
        val ratio = trainTokens.toDouble() / total
        assertTrue(ratio in 0.89..0.91, "train 비율이 ~90%여야 했으나 ${ratio}")
    }

    @Test
    fun separateInputsPreferredOverStories() {
        // 세 파일이 모두 있으면 분리 경로가 우선.
        val dir = File(root, "all-three").apply { mkdirs() }
        File(dir, "train.txt").writeText("alpha ".repeat(500))
        File(dir, "val.txt").writeText("beta ".repeat(100))
        File(dir, "stories.txt").writeText("gamma ".repeat(10000))  // 이게 쓰이면 토큰 수가 폭증

        StoriesBPEPrep.run(dir.absolutePath, verbose = false)

        val meta = Json { ignoreUnknownKeys = true }
            .decodeFromString<MetaInfo>(File(dir, "meta.json").readText())
        // train.txt 선호 — 'g'(gamma) 문자는 vocab에 없어야.
        val chars = meta.stringToIndex.keys.flatMap { it.toList() }.toSet()
        assertTrue('g' !in chars, "stories.txt의 gamma 문자가 vocab에 섞이면 분리 경로가 무시된 것")
    }

    private fun readTokens(file: File): Int {
        val bytes = file.readBytes()
        return bytes.size / 4
    }
}
