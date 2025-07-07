import kotlin.math.*
import kotlin.random.Random


class RandomGaussian {
    companion object {

        // 이미 생성된 가우시안 난수가 있는지 여부를 저장하는 상태 변수
        private var haveNextNextGaussian = false

        // 다음에 반환할 가우시안 난수를 저장하는 변수
        private var nextNextGaussian = 0.0

        /**
         * Random.Default 객체에 대한 확장 함수로 nextGaussian()을 구현합니다.
         * 이 함수는 평균 0.0, 표준편차 1.0의 가우시안(정규) 분포를 따르는 double 값을 반환합니다.
         *
         * @return 표준 정규 분포를 따르는 난수
         */
        fun next(): Double {
            // 이전에 계산해 둔 난수가 있다면, 그 값을 반환하고 상태를 초기화합니다.
            if (haveNextNextGaussian) {
                haveNextNextGaussian = false
                return nextNextGaussian
            }

            // 이전에 계산한 값이 없다면, 박스-뮬러 변환을 사용하여 새로 생성합니다.
            var u1: Double
            var u2: Double

            // u1이 0이 되면 log(u1)이 무한대가 되므로, 0이 아닌 값이 나올 때까지 반복합니다.
            do {
                u1 = Random.nextDouble() // 0.0(포함)과 1.0(제외) 사이의 균등 분포 난수
            } while (u1 == 0.0)
            u2 = Random.nextDouble()

            // 박스-뮬러 변환 공식 적용
            val z0 = sqrt(-2.0 * log(u1, E)) * cos(2.0 * PI * u2)
            val z1 = sqrt(-2.0 * log(u1, E)) * sin(2.0 * PI * u2)
            // 참고: log(u1)은 자연로그(밑이 e)입니다. Kotlin에서는 log(x, E) 또는 ln(x)를 사용합니다.

            // z1을 다음 호출을 위해 저장해 둡니다.
            nextNextGaussian = z1
            haveNextNextGaussian = true

            // z0을 이번 호출의 결과로 반환합니다.
            return z0
        }
    }

}
