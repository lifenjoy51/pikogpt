import kotlin.test.Test

class ValueTest {

    // 사용 예제
    @Test
    fun main() {
        // 간단한 연산 그래프 생성
        val a = Value(2.0f)
        val b = Value(3.0f)
        val c = a * b + b.pow(2.0f)
        val d = c.relu()

        println("Forward pass:")
        println("a = $a")
        println("b = $b")
        println("c = $c")
        println("d = $d")

        // 역전파 수행
        d.backward()

        println("\nAfter backward pass:")
        println("a = $a")
        println("b = $b")
        println("c = $c")
        println("d = $d")
    }
}