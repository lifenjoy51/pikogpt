

/**
 * `Iterable`의 각 요소에 대해 주어진 [selector] 함수를 적용한 결과(Float)의 합계를 반환합니다.
 *
 * 이 확장 함수는 표준 라이브러리의 `sumOf`가 Double, Int, Long 등 기본 숫자 타입만 지원하는 것을 보완하여,
 * Float 타입에 대한 합계를 효율적으로 계산할 수 있도록 합니다.
 *
 * @param selector 각 요소에 적용할 변환 함수. Float 값을 반환해야 합니다.
 * @return 모든 요소에 대한 [selector] 적용 결과의 합계 (Float).
 */
fun <T> Iterable<T>.sumOf(selector: (T) -> Float): Float {
    var sum = 0f
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * `Array`의 각 요소에 대해 주어진 [selector] 함수를 적용한 결과(Float)의 합계를 반환합니다.
 *
 * 이 확장 함수는 표준 라이브러리의 `sumOf`가 Double, Int, Long 등 기본 숫자 타입만 지원하는 것을 보완하여,
 * Float 타입에 대한 합계를 효율적으로 계산할 수 있도록 합니다.
 *
 * @param selector 각 요소에 적용할 변환 함수. Float 값을 반환해야 합니다.
 * @return 모든 요소에 대한 [selector] 적용 결과의 합계 (Float).
 */
fun <T> Array<out T>.sumOf(selector: (T) -> Float): Float {
    var sum = 0f
    for (element in this) {
        sum += selector(element)
    }
    return sum
}