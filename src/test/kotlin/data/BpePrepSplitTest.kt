package data

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * `BpePrep.run()`의 분리 입력 경로를 검증.
 *
 * 입력 규약: `train.txt` + `val.txt` 둘 다 있어야 하며, BPE는 train에서만 학습 (val leakage 방지).
 */
class BpePrepSplitTest {
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

        BpePrep.run(dir.absolutePath, verbose = false)

        val meta = Json { ignoreUnknownKeys = true }
            .decodeFromString<MetaInfo>(File(dir, "meta.json").readText())
        val chars = meta.stringToIndex.keys.flatMap { it.toList() }.toSet()
        // train에만 a/b/c, val에만 z/y — BPE 학습이 train에서만 됐으면 z/y는 단일 문자로도 등장 안 함.
        assertTrue('a' in chars || 'b' in chars || 'c' in chars, "train 글자 중 하나는 vocab에 있어야")
        assertTrue('z' !in chars && 'y' !in chars, "val-only 글자는 vocab에 없어야 (leakage 없음)")

        val trainTokens = readTokens(File(dir, "train.bin"))
        val valTokens = readTokens(File(dir, "val.bin"))
        assertTrue(trainTokens > 0 && valTokens > 0, "bin 파일 모두 생성되어야")
    }

    @Test
    fun missingValTxtFails() {
        val dir = File(root, "no-val").apply { mkdirs() }
        File(dir, "train.txt").writeText("alpha ".repeat(100))

        try {
            BpePrep.run(dir.absolutePath, verbose = false)
            assertTrue(false, "val.txt가 없으면 예외가 발생해야 함")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("train.txt 와 val.txt") == true, "에러 메시지 확인")
        }
    }

    private fun readTokens(file: File): Int {
        val bytes = file.readBytes()
        return bytes.size / 4
    }
}
