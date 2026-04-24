package data

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * SimpleBPE의 새 기능(lowercase, word pre-tokenize, merges 직렬화/복원)을 검증.
 */
class SimpleBPEFeaturesTest {

    @Test
    fun lowercaseNormalizesBeforeTraining() {
        val bpe = SimpleBPE(maxVocabSize = 50, lowercase = true, verbose = false)
        bpe.train("Hello HELLO hello HeLLo")
        val stoi = bpe.getStoi()
        // 소문자 플래그 on이면 대문자 토큰이 어휘에 들어오지 않아야 함
        assertTrue("H" !in stoi, "lowercase=true에서 'H'가 어휘에 있으면 안 됨")
        assertTrue("E" !in stoi, "'E'도 마찬가지")
        assertTrue("h" in stoi && "e" in stoi, "소문자 문자는 있어야 함")
    }

    @Test
    fun wordPreTokenizeKeepsWordBoundariesSeparate() {
        val bpe = SimpleBPE(
            maxVocabSize = 200,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        bpe.train("the cat sat on the mat the cat")
        val stoi = bpe.getStoi()
        // 단어 경계를 넘는 토큰 (예: "t t" 또는 "t c")은 만들어지면 안 됨
        val invalid = stoi.keys.filter { token ->
            // 공백이 중간에 있고 앞뒤로 문자가 있으면 경계 넘음
            val trimmed = token.trim()
            trimmed.contains(' ') && trimmed.isNotEmpty()
        }
        assertTrue(
            invalid.isEmpty(),
            "단어 경계를 넘는 토큰이 있으면 안 됨: $invalid"
        )
    }

    @Test
    fun encodeWithPreTokenizeIsConsistent() {
        val bpe = SimpleBPE(
            maxVocabSize = 100,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        bpe.train("the quick brown fox jumps over the lazy dog")
        val ids1 = bpe.encode("the quick brown fox")
        val ids2 = bpe.encode("the quick brown fox")
        assertEquals(ids1, ids2, "같은 입력은 같은 토큰 ID 시퀀스를 내야 함")
    }

    @Test
    fun restoreReproducesSameEncoding() {
        // 원본 학습
        val original = SimpleBPE(
            maxVocabSize = 120,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        original.train("Once upon a time there was a little cat. The cat was very happy.")

        val stoi = original.getStoi()
        val merges = original.getMerges()

        // 빈 인스턴스로 복원
        val restored = SimpleBPE(
            maxVocabSize = 120,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        restored.restore(stoi, merges)

        val sample = "once upon a time the cat was happy"
        val ids1 = original.encode(sample)
        val ids2 = restored.encode(sample)
        assertEquals(ids1, ids2, "복원된 인스턴스가 원본과 동일한 토큰 시퀀스를 재생해야 함")
    }
}
