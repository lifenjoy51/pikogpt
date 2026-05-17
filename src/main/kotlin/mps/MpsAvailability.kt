package mps

import java.io.File

/**
 * macOS + arm64 + libpikogpt_metal.dylib 로드 + Metal 초기화 가능성을 한 번 평가.
 *
 * 호출 측은 `MpsAvailability.available`만 보고 분기. dylib이 없거나 device 초기화에 실패하면
 * `false`로 굳어지고 이후엔 turbo CPU 경로로 fallback.
 */
object MpsAvailability {

    @Volatile private var checked = false
    @Volatile var available: Boolean = false
        private set
    @Volatile var reason: String = "not checked"
        private set

    @Synchronized
    fun ensureChecked(): Boolean {
        if (checked) return available
        checked = true

        val osName = System.getProperty("os.name").lowercase()
        val osArch = System.getProperty("os.arch").lowercase()
        if (!osName.contains("mac")) {
            reason = "not macOS (os.name=$osName)"
            return false
        }
        if (osArch != "aarch64" && osArch != "arm64") {
            reason = "not arm64 (os.arch=$osArch)"
            return false
        }

        val libFile = resolveDylib()
        if (libFile == null || !libFile.exists()) {
            reason = "libpikogpt_metal.dylib not found (run ./gradlew buildMetalLib)"
            return false
        }

        try {
            System.load(libFile.absolutePath)
        } catch (t: Throwable) {
            reason = "System.load failed: ${t.message}"
            return false
        }

        val initOk = try {
            mps.jni.MetalMatMulBridge.nativeInit()
        } catch (t: Throwable) {
            reason = "nativeInit threw: ${t.message}"
            return false
        }
        if (!initOk) {
            reason = "nativeInit returned false (Metal device/pipeline init failed)"
            return false
        }

        available = true
        reason = "ok (loaded ${libFile.absolutePath})"
        return true
    }

    private fun resolveDylib(): File? {
        // 1) -Djava.library.path 우선
        System.getProperty("java.library.path")?.split(File.pathSeparator)?.forEach { dir ->
            val f = File(dir, "libpikogpt_metal.dylib")
            if (f.exists()) return f
        }
        // 2) build/native (gradle 기본 출력)
        val cwdDefault = File("build/native/libpikogpt_metal.dylib")
        if (cwdDefault.exists()) return cwdDefault
        return null
    }
}
