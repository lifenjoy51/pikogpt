package data

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Default specialTokens 배치와 문서 경계(bos/eos) 토큰이 학습·인코딩에서
 * 예상대로 동작하는지 확인.
 */
class CharBPESpecialsTest {

    @Test
    fun defaultSpecialsAreAtFixedIds() {
        val bpe = CharBPE(maxVocabSize = 20, verbose = false)
        bpe.train("hello")  // 아무 텍스트라도 OK
        val stoi = bpe.getStoi()
        assertEquals(0, stoi["<|eos|>"], "<|eos|>는 기본 배치에서 id 0")
        assertEquals(1, stoi["<|unk|>"], "<|unk|>는 id 1")
        assertEquals(2, stoi["<|bos|>"], "<|bos|>는 id 2")
    }

    @Test
    fun bosEosPreservedThroughEncoding() {
        // CharBPE는 special 토큰 주변에 공백을 붙여 단일 토큰으로 잡기 때문에,
        // 인코딩 결과의 정확한 첫/끝 ID는 구현 세부사항이다. 여기서는 bos와 eos가
        // **결과 시퀀스에 반드시 포함**되는지만 검증한다.
        val bpe = CharBPE(
            maxVocabSize = 50,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        bpe.train("<|bos|>hello world<|eos|>")
        val ids = bpe.encode("<|bos|>hello<|eos|>")

        assertTrue(ids.isNotEmpty())
        assertTrue(2 in ids, "bos(id 2)가 인코딩 결과에 포함되어야 함")
        assertTrue(0 in ids, "eos(id 0)가 인코딩 결과에 포함되어야 함")
    }

    @Test
    fun specialTokensNotMerged() {
        // 병합 루프가 돌아도 special 토큰은 인접 토큰과 merge되지 않아야 함
        val bpe = CharBPE(maxVocabSize = 100, verbose = false)
        bpe.train("<|bos|>ababab<|eos|>".repeat(10))
        val stoi = bpe.getStoi()
        // vocab에 "<|bos|>a"나 "b<|eos|>" 같은 잘못된 merge가 없어야 함
        val bosBad = stoi.keys.filter { it.contains("<|bos|>") && it != "<|bos|>" }
        val eosBad = stoi.keys.filter { it.contains("<|eos|>") && it != "<|eos|>" }
        val bad = bosBad + eosBad
        assertTrue(bad.isEmpty(), "특수 토큰이 포함된 merge가 있으면 안 됨: $bad")
    }

    @Test
    fun restoreRoundTripPreservesSpecials() {
        val original = CharBPE(
            maxVocabSize = 80,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        original.train("<|bos|>the cat sat<|eos|>")

        val restored = CharBPE(
            maxVocabSize = 80,
            lowercase = true,
            useWordPreTokenize = true,
            verbose = false,
        )
        restored.restore(original.getStoi(), original.getMerges())

        val sample = "<|bos|>the cat<|eos|>"
        assertEquals(
            original.encode(sample),
            restored.encode(sample),
            "restore된 인스턴스가 동일한 id 시퀀스를 내야 함 (특수 토큰 포함)"
        )
    }
}
