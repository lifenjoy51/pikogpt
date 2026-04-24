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
// 규약: 모든 실행 스크립트는 main 소스셋에 있고, JavaExec 태스크로 실행됨.

tasks.register<JavaExec>("runMain") {
    description = "Run MainKt"
    mainClass.set("MainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runStoriesBpe") {
    description = "Run StoriesBpePrep (스토리 BPE 처리)"
    mainClass.set("data.StoriesBpePrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runAlphabetPrep") {
    description = "Run AlphabetPrep (알파벳 데이터 준비)"
    mainClass.set("data.AlphabetPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runStoryGenerator") {
    description = "Run StoryGenerator (스토리 생성)"
    mainClass.set("data.StoryGeneratorKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runBPETest") {
    description = "Run BPETest (BPE 테스트)"
    mainClass.set("BPETestKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runTrainer") {
    description = "Run TrainerMain (학습 smoke)"
    mainClass.set("train.TrainerMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runMiniTrainer") {
    description = "Run MiniTrainerMain (경량 학습 - az 알파벳)"
    mainClass.set("train.MiniTrainerMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runTinyHelenTrain") {
    description = "Run TinyHelenTrain (TinyHelen leaner 코퍼스 overnight 학습 — 스칼라 백엔드)"
    mainClass.set("train.TinyHelenTrainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runTinyHelenTrainVec") {
    description = "Run TinyHelenTrainVec (~1M 파라미터, 벡터 백엔드)"
    mainClass.set("train.TinyHelenTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runSampler") {
    description = "Run SamplerMain (샘플링)"
    mainClass.set("sample.SamplerMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runTinyHelenSample") {
    description = "Run TinyHelenSample (model/ 최신 스칼라 체크포인트 자동 샘플링)"
    mainClass.set("sample.TinyHelenSampleKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runTinyHelenSampleVec") {
    description = "Run TinyHelenSampleVec (model/vec/ 최신 벡터 체크포인트 자동 샘플링)"
    mainClass.set("sample.TinyHelenSampleVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runAnalyzeTokens") {
    description = "Run AnalyzeTokensMain (토큰 분포 분석)"
    mainClass.set("data.AnalyzeTokensMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runDebugBPE") {
    description = "Run DebugBPEMain (BPE 디버깅)"
    mainClass.set("data.DebugBPEMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}
