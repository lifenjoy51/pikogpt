plugins {
    kotlin("jvm") version "1.9.0"
    kotlin("plugin.serialization") version "1.9.0"
}

group = "me.lifenjoy51"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
}

tasks.test {
    useJUnitPlatform()
}

// Main 함수들을 실행하는 Gradle 태스크들
tasks.register("runMain") {
    dependsOn("classes")
    doLast {
        exec {
            commandLine("java", "-Xmx2g", "-cp", "${sourceSets.main.get().runtimeClasspath.asPath}", "MainKt")
        }
    }
}

tasks.register("runStoriesBpe") {
    description = "Run StoriesBpePrep (스토리 BPE 처리)"
    dependsOn("classes")
    doLast {
        exec {
            commandLine("java", "-Xmx4g", "-cp", "${sourceSets.main.get().runtimeClasspath.asPath}", "data.StoriesBpePrepKt")
        }
    }
}

tasks.register("runAlphabetPrep") {
    description = "Run AlphabetPrep (알파벳 데이터 준비)"
    dependsOn("classes")
    doLast {
        exec {
            commandLine("java", "-Xmx2g", "-cp", "${sourceSets.main.get().runtimeClasspath.asPath}", "data.AlphabetPrepKt")
        }
    }
}

tasks.register("runStoryGenerator") {
    description = "Run StoryGenerator (스토리 생성)"
    dependsOn("classes")
    doLast {
        exec {
            commandLine("java", "-Xmx4g", "-cp", "${sourceSets.main.get().runtimeClasspath.asPath}", "data.StoryGeneratorKt")
        }
    }
}

tasks.register("runBPETest") {
    description = "Run BPETest (BPE 테스트)"
    dependsOn("classes")
    doLast {
        exec {
            commandLine("java", "-Xmx2g", "-cp", "${sourceSets.main.get().runtimeClasspath.asPath}", "BPETestKt")
        }
    }
}