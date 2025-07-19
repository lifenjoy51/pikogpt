package data

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SimpleBPETest {

    @Test
    fun testBasicTrainingAndEncoding() {
        val bpe = SimpleBPE(maxVocabSize = 100)
        val text = "hello world hello"

        bpe.train(text)

        assertTrue(bpe.getVocabSize() > 0)
        assertTrue(bpe.getVocabSize() <= 100)

        val encoded = bpe.encode(text)
        assertTrue(encoded.isNotEmpty())
    }

    @Test
    fun testSpecialTokens() {
        val specialTokens = listOf("<|eos|>", " ", "<|start|>")
        val bpe = SimpleBPE(maxVocabSize = 50, specialTokens = specialTokens)
        val text = "hello<|eos|>world<|start|>"

        bpe.train(text)

        val vocab = bpe.getStoi()
        specialTokens.forEach { token ->
            assertTrue(vocab.containsKey(token), "Special token '$token' should be in vocabulary")
        }
    }

    @Test
    fun testEmptyTextEncoding() {
        val bpe = SimpleBPE(maxVocabSize = 50)
        bpe.train("hello world")

        val encoded = bpe.encode("")
        assertTrue(encoded.isEmpty())
    }

    @Test
    fun testSingleCharacterText() {
        val bpe = SimpleBPE(maxVocabSize = 50)
        val text = "a"

        bpe.train(text)
        val encoded = bpe.encode(text)

        assertEquals(1, encoded.size)
    }

    @Test
    fun testRepeatedCharacters() {
        val bpe = SimpleBPE(maxVocabSize = 100)
        val text = "aaabbbccc"

        bpe.train(text)
        val encoded = bpe.encode(text)

        assertTrue(encoded.isNotEmpty())
        assertTrue(encoded.size <= text.length)
    }

    @Test
    fun testVocabularyMapping() {
        val bpe = SimpleBPE(maxVocabSize = 50)
        val text = "abc"

        bpe.train(text)

        val stoi = bpe.getStoi()
        val itos = bpe.getItos()

        assertEquals(stoi.size, itos.size)

        stoi.forEach { (token, id) ->
            assertEquals(token, itos[id])
        }
    }

    @Test
    fun testEncodingConsistency() {
        val bpe = SimpleBPE(maxVocabSize = 100)
        val text = "hello world test"

        bpe.train(text)

        val encoded1 = bpe.encode(text)
        val encoded2 = bpe.encode(text)

        assertEquals(encoded1, encoded2)
    }

    @Test
    fun testLargerVocabularySize() {
        val bpe = SimpleBPE(maxVocabSize = 1000)
        val text = "the quick brown fox jumps over the lazy dog".repeat(10)

        bpe.train(text)

        assertTrue(bpe.getVocabSize() > 50)

        val encoded = bpe.encode(text)
        assertTrue(encoded.isNotEmpty())
    }

    @Test
    fun testSpecialTokensInEncoding() {
        val bpe = SimpleBPE(maxVocabSize = 100, specialTokens = listOf("<|eos|>", " "))
        val text = "hello <|eos|> world"

        bpe.train(text)
        val encoded = bpe.encode(text)

        assertTrue(encoded.isNotEmpty())

        val vocab = bpe.getStoi()
        assertTrue(vocab.containsKey("<|eos|>"))
    }

    @Test
    fun testUnknownTokenHandling() {
        val bpe = SimpleBPE(maxVocabSize = 50)
        bpe.train("abc")

        val encodedKnown = bpe.encode("abc")
        val encodedUnknown = bpe.encode("xyz")

        assertTrue(encodedKnown.isNotEmpty())
        assertTrue(encodedUnknown.isNotEmpty())
    }

    @Test
    fun testMinimalVocabSize() {
        val bpe = SimpleBPE(maxVocabSize = 5)
        val text = "ab"

        bpe.train(text)

        assertTrue(bpe.getVocabSize() <= 5)
        assertTrue(bpe.getVocabSize() >= 2) // At least special tokens
    }

    @Test
    fun testCompressionRatio() {
        val bpe = SimpleBPE(maxVocabSize = 200)
        val text = "hello hello hello world world world"

        bpe.train(text)
        val encoded = bpe.encode(text)

        assertTrue(encoded.size < text.length, "BPE should compress repeated patterns")
    }

    @Test
    fun testTokenPairMerging() {
        val bpe = SimpleBPE(maxVocabSize = 100)
        val text = "abab"

        bpe.train(text)

        val vocab = bpe.getStoi()
        assertTrue(vocab.containsKey("a"))
        assertTrue(vocab.containsKey("b"))

        val encoded = bpe.encode(text)
        assertTrue(encoded.size <= 4)
    }
}