import kotlin.math.*
import kotlin.random.Random

/**
 * 표준 정규 분포(가우시안) 난수를 생성하는 유틸리티 클래스입니다.
 *
 * 이 클래스는 박스-뮬러 변환(Box-Muller transform) 알고리즘을 사용하여
 * 균등 분포 난수로부터 정규 분포 난수를 생성합니다.
 * 한 번의 변환으로 두 개의 독립적인 정규 분포 난수를 생성할 수 있으므로,
 * 하나는 즉시 반환하고 다른 하나는 다음 호출을 위해 저장하여 효율성을 높입니다.
 */
class RandomGaussian {
    companion object {
        /** 다음에 반환할 가우시안 난수가 저장되어 있는지 여부를 나타내는 플래그 */
        private var haveNextNextGaussian = false

        /** 다음 호출 시 반환하기 위해 저장해 둔 가우시안 난수 */
        private var nextNextGaussian = 0.0

        /**
         * 평균 0.0, 표준편차 1.0의 표준 정규 분포를 따르는 double 타입의 난수를 반환합니다.
         *
         * 박스-뮬러 변환 알고리즘:
         * 1. 0과 1 사이의 균등 분포 난수 두 개(u1, u2)를 생성합니다.
         * 2. 이들을 사용하여 독립적인 표준 정규 분포 난수 두 개(z0, z1)를 계산합니다.
         *    - z0 = sqrt(-2 * ln(u1)) * cos(2 * π * u2)
         *    - z1 = sqrt(-2 * ln(u1)) * sin(2 * π * u2)
         * 3. z0을 현재 호출의 결과로 반환하고, z1은 다음 호출을 위해 저장합니다.
         * 4. 다음 호출 시에는 저장된 z1을 즉시 반환하여 계산을 절약합니다.
         *
         * @return 표준 정규 분포를 따르는 난수 (Double)
         */
        fun next(): Double {
            // 이전에 계산해 둔 난수가 있다면, 그 값을 반환하고 상태를 초기화합니다.
            if (haveNextNextGaussian) {
                haveNextNextGaussian = false
                return nextNextGaussian
            }

            // 박스-뮬러 변환을 위한 균등 분포 난수 생성
            var u1: Double
            do {
                u1 = Random.nextDouble() // log(0)은 -Infinity이므로 u1이 0이 아니어야 함
            } while (u1 == 0.0)
            val u2 = Random.nextDouble()

            // 박스-뮬러 변환 공식 적용
            val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
            val z1 = sqrt(-2.0 * ln(u1)) * sin(2.0 * PI * u2)

            // z1을 다음 호출을 위해 저장
            nextNextGaussian = z1
            haveNextNextGaussian = true

            // z0을 현재 결과로 반환
            return z0
        }
    }
}
