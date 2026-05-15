import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

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

    // 추론 API 서버용 (server.InferenceApiMain)
    implementation("io.ktor:ktor-server-core:2.3.12")
    implementation("io.ktor:ktor-server-netty:2.3.12")
    implementation("io.ktor:ktor-server-content-negotiation:2.3.12")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.12")
}

// turbo 백엔드용 JDK 21 toolchain + Java Vector API (jdk.incubator.vector).
// Kotlin 1.9.0은 jvmTarget=21을 지원하지 않으므로 컴파일 target은 17로 유지.
// Toolchain은 JDK 21 — Vector API는 JDK 21 런타임에서 활성화 (--add-modules).
kotlin {
    jvmToolchain(21)
}

tasks.withType<KotlinCompile>().configureEach {
    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs = freeCompilerArgs + listOf(
            "-Xadd-modules=jdk.incubator.vector",
        )
    }
}

tasks.withType<JavaCompile>().configureEach {
    // Kotlin 1.9.0의 jvmTarget=17과 일치시켜 compile target 호환성 확보.
    sourceCompatibility = "17"
    targetCompatibility = "17"
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--add-modules=jdk.incubator.vector")
}

// Main 함수들을 실행하는 Gradle 태스크들
// 규약: 모든 실행 스크립트는 main 소스셋에 있고, JavaExec 태스크로 실행됨.

tasks.register<JavaExec>("runMain") {
    description = "Run MainKt"
    mainClass.set("MainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runBpe") {
    description = "Run BpePrep (BPE 학습 + 인코딩)"
    mainClass.set("data.BpePrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx12g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runAlphabetPrep") {
    description = "Run AlphabetPrep — data/alphabet/az.txt를 토큰화해 train.bin/val.bin/meta.json 생성"
    mainClass.set("data.AlphabetPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runBPETest") {
    description = "Run BPETest (BPE 테스트)"
    mainClass.set("BPETestKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runMiniTrainer") {
    description = "Run MiniTrainerMain — Scalar 백엔드 quickstart 학습 (data/alphabet, ~10분)"
    mainClass.set("train.MiniTrainerMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx2g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
        "-XX:ActiveProcessorCount=8",
        "-Djava.util.concurrent.ForkJoinPool.common.parallelism=8",
    )
}

tasks.register<JavaExec>("runTinyHelenTrain") {
    description = "Run TinyHelenTrain (TinyHelen leaner 코퍼스 overnight 학습 — 스칼라 백엔드)"
    mainClass.set("train.experiments.TinyHelenTrainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTinyHelenTrainTextbookTurbo") {
    description = "Run TinyHelenTrainTextbookTurbo (textbook-only, ~1M 파라미터)"
    mainClass.set("train.experiments.TinyHelenTrainTextbookTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTinyHelenTrainConversationTurbo") {
    description = "Run TinyHelenTrainConversationTurbo (100M conversation, ~1M 파라미터, 12k iter)"
    mainClass.set("train.experiments.TinyHelenTrainConversationTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixTrainTurbo") {
    description = "Run ConvMixTrainTurbo (TinyHelen conv + TinyDialogues age-5 stripped, ~1M 파라미터)"
    mainClass.set("train.experiments.ConvMixTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixTurnTrainTurbo") {
    description = "Run ConvMixTurnTrainTurbo (conv-mix + <|turn|> 토큰, 432k)"
    mainClass.set("train.experiments.ConvMixTurnTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixTurnNoqTrainTurbo") {
    description = "Run ConvMixTurnNoqTrainTurbo (conv-mix-turn 따옴표 제거, 432k)"
    mainClass.set("train.experiments.ConvMixTurnNoqTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixTurnNoqB128TrainTurbo") {
    description = "Run ConvMixTurnNoqB128TrainTurbo (blockSize 128, 436k params)"
    mainClass.set("train.experiments.ConvMixTurnNoqB128TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixA510TrainTurbo") {
    description = "Run ConvMixA510TrainTurbo (TinyHelen + age-5 + age-10, ~18M tok, 432k)"
    mainClass.set("train.experiments.ConvMixA510TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixA510M773TrainTurbo") {
    description = "Run ConvMixA510M773TrainTurbo (a510 데이터 + 773k tied 모델, 12k iter)"
    mainClass.set("train.experiments.ConvMixA510M773TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixA510M773B128TrainTurbo") {
    description = "Run ConvMixA510M773B128TrainTurbo (773k tied + blockSize 128, 8k iter)"
    mainClass.set("train.experiments.ConvMixA510M773B128TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixCleanA510M773TrainTurbo") {
    description = "Run ConvMixCleanA510M773TrainTurbo (clean 데이터 + 773k tied 모델, 12k iter)"
    mainClass.set("train.experiments.ConvMixCleanA510M773TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixCleanA510M773SwiGLUTrainTurbo") {
    description = "Run ConvMixCleanA510M773SwiGLUTrainTurbo (clean + 773k tied + SwiGLU MLP)"
    mainClass.set("train.experiments.ConvMixCleanA510M773SwiGLUTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runConvMixCleanA510M773SwiGLURoPETrainTurbo") {
    description = "Run ConvMixCleanA510M773SwiGLURoPETrainTurbo (clean + tied + SwiGLU + RoPE)"
    mainClass.set("train.experiments.ConvMixCleanA510M773SwiGLURoPETrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runDialoguesA510M773SwiGLURoPETrainTurbo") {
    description = "Run DialoguesA510M773SwiGLURoPETrainTurbo (TinyDialogues age-5+10 only, tied + SwiGLU + RoPE)"
    mainClass.set("train.experiments.DialoguesA510M773SwiGLURoPETrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTwoStageBaseTrainTurbo") {
    description = "Run TwoStageBaseTrainTurbo (TinyHelen wiki+textbook BASE pretrain — 사실 지식 주입)"
    mainClass.set("train.experiments.TwoStageBaseTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTwoStageITTrainTurbo") {
    description = "Run TwoStageITTrainTurbo (dialogues-a510 IT finetune + 20% BASE replay)"
    mainClass.set("train.experiments.TwoStageITTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTwoStageBaseV2TrainTurbo") {
    description = "Run TwoStageBaseV2TrainTurbo (TinyHelen wiki+textbook+web+book BASE pretrain, vocab 2000)"
    mainClass.set("train.experiments.TwoStageBaseV2TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTwoStageITV2TrainTurbo") {
    description = "Run TwoStageITV2TrainTurbo (v2 IT finetune + 20% v2 BASE replay)"
    mainClass.set("train.experiments.TwoStageITV2TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runThreeStageDictTrainV4Turbo") {
    description = "Run ThreeStageDictTrainV4Turbo (Stage 1: dict scratch pretrain, ~2.5k iter, 12 epoch)"
    mainClass.set("train.experiments.ThreeStageDictTrainV4TurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runThreeStageWikiTrainV4Turbo") {
    description = "Run ThreeStageWikiTrainV4Turbo (Stage 2: wiki finetune from dict ckpt + 30% dict replay)"
    mainClass.set("train.experiments.ThreeStageWikiTrainV4TurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runThreeStageConvTrainV4Turbo") {
    description = "Run ThreeStageConvTrainV4Turbo (Stage 3: conv finetune from wiki ckpt + multi-replay 15% dict + 15% wiki)"
    mainClass.set("train.experiments.ThreeStageConvTrainV4TurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runThreeStageDictTrainV5Turbo") {
    description = "Run ThreeStageDictTrainV5Turbo (Stage 1 v5: dict scratch, emb 144 / L9 / H6, ~2.5M params)"
    mainClass.set("train.experiments.ThreeStageDictTrainV5TurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runThreeStageWikiTrainV5Turbo") {
    description = "Run ThreeStageWikiTrainV5Turbo (Stage 2 v5: wiki finetune from dict ckpt, 30% dict replay)"
    mainClass.set("train.experiments.ThreeStageWikiTrainV5TurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runThreeStageConvTrainV5Turbo") {
    description = "Run ThreeStageConvTrainV5Turbo (Stage 3 v5: conv finetune from wiki ckpt, 15%+15% multi-replay)"
    mainClass.set("train.experiments.ThreeStageConvTrainV5TurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV2ProStage1TrainTurbo") {
    description = "Run CcmcV2ProStage1TrainTurbo (CCMC v2-pro Stage 1: binding scratch, 8L×96D×3H SwiGLU+RoPE, vocab 2000)"
    mainClass.set("train.experiments.CcmcV2ProStage1TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runEncodeWithExistingMeta") {
    description = "Run EncodeWithExistingMeta (공유 meta.json으로 다른 디렉터리 인코딩)"
    mainClass.set("data.EncodeWithExistingMetaKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx12g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runSplitByTokenRatio") {
    description = "Run SplitByTokenRatio (record-per-line 입력을 토큰 수 기준으로 train/val 분할)"
    mainClass.set("data.SplitByTokenRatioKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runSampler") {
    description = "Run SamplerMain — Scalar quickstart 샘플링 (인자 없으면 model/alphabet/main 최신 v 자동 검색)"
    mainClass.set("sample.SamplerMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runTinyHelenSample") {
    description = "Run TinyHelenSample (model/ 최신 스칼라 체크포인트 자동 샘플링)"
    mainClass.set("sample.TinyHelenSampleKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

// turbo 백엔드 — Phase 0~5 진행 중. JDK 21 + Java Vector API 활용.
tasks.register<JavaExec>("runTinyHelenTrainTurbo") {
    description = "Run TinyHelenTrainTurbo (~1M 파라미터, turbo 백엔드)"
    mainClass.set("train.experiments.TinyHelenTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runWordStart4PrefixTrainTurbo") {
    description = "Run WordStart4PrefixTrainTurbo (~300k 파라미터, turbo 백엔드, 12코어/worker 4)"
    mainClass.set("train.experiments.WordStart4PrefixTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx4g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
        "-XX:ActiveProcessorCount=12",
        "-Djava.util.concurrent.ForkJoinPool.common.parallelism=12",
    )
    environment("TURBO_MAX_WORKERS", "4")
}

tasks.register<JavaExec>("runCcmcAllV2048WiderTrainTurbo") {
    description = "Run CcmcAllV2048WiderTrainTurbo (~485k 파라미터, v6 corpus, turbo 백엔드)"
    mainClass.set("train.experiments.CcmcAllV2048WiderTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx4g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
        "-XX:ActiveProcessorCount=12",
        "-Djava.util.concurrent.ForkJoinPool.common.parallelism=12",
    )
    environment("TURBO_MAX_WORKERS", "4")
}

tasks.register<JavaExec>("runCcmcAllV2048WiderH2TrainTurbo") {
    description = "Run CcmcAllV2048WiderH2TrainTurbo (~485k 파라미터, v7 corpus + BOS/EOS, heads=2, turbo, workers=8)"
    mainClass.set("train.experiments.CcmcAllV2048WiderH2TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx4g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
    environment("TURBO_MAX_WORKERS", "8")
}

tasks.register<JavaExec>("runCcmcAllV2048M1TrainTurbo") {
    description = "Run CcmcAllV2048M1TrainTurbo (~986k 파라미터, v8 = v7 data + 1M 모델, heads=3, turbo, workers=8)"
    mainClass.set("train.experiments.CcmcAllV2048M1TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx6g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
    environment("TURBO_MAX_WORKERS", "8")
}

tasks.register<JavaExec>("runCcmcLemmaV1024TrainTurbo") {
    description = "ccmc-lemma-v1024 학습 (d=32 heads=2 L=5 blockSize=32, ~97k params, 10k iter)"
    mainClass.set("train.experiments.CcmcLemmaV1024TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx2g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
    environment("TURBO_MAX_WORKERS", "4")
}

tasks.register<JavaExec>("runCcmcAllCompareElfTrainTurbo") {
    description = "ELF vs pikogpt 비교용 학습 (Wider 485k @ v6, 10k iter, expName=compare-vs-elf)"
    mainClass.set("train.experiments.CcmcAllCompareElfTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx4g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
    environment("TURBO_MAX_WORKERS", "4")
}

tasks.register<JavaExec>("runCompareSampler") {
    description = "ELF vs pikogpt 비교용 turbo 샘플링 (compare-vs-elf ckpt에서 BOS 시작 생성)"
    mainClass.set("sample.CompareSamplerMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx4g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
}

tasks.register<JavaExec>("runCcmcAllV4096M1TrainTurbo") {
    description = "Run CcmcAllV4096M1TrainTurbo (~1.07M 파라미터, v9 = vocab=4096 + 1M 모델 L=6 H=3, turbo, workers=4)"
    mainClass.set("train.experiments.CcmcAllV4096M1TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx6g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
    environment("TURBO_MAX_WORKERS", "4")
}

tasks.register<JavaExec>("runCcmcAllV4096M1LemmaW10TrainTurbo") {
    description = "Run CcmcAllV4096M1LemmaW10TrainTurbo (~1.07M 파라미터, v10 = v9 + lemma stream weight 0.1, turbo, workers=4)"
    mainClass.set("train.experiments.CcmcAllV4096M1LemmaW10TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx6g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
    )
    environment("TURBO_MAX_WORKERS", "4")
}

tasks.register<JavaExec>("runTinyHelenSampleTurbo") {
    description = "Run TinyHelenSampleTurbo (model/<dataset>/turbo/ 최신 체크포인트 자동 샘플링)"
    mainClass.set("sample.TinyHelenSampleTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector")
}

tasks.register<JavaExec>("runTurboBench") {
    description = "Run TurboMicroBench (turbo MatMul / AdamW 벤치마크 vs vec)"
    mainClass.set("turbo.bench.TurboMicroBenchKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector")
}

tasks.register<JavaExec>("runCcmcV2ProStage2TrainTurbo") {
    description = "Run CcmcV2ProStage2TrainTurbo (Stage 2 instruction finetune, turbo 백엔드)"
    mainClass.set("train.experiments.CcmcV2ProStage2TrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runBench10MTurbo") {
    description = "Run Bench10MTurbo — 10M params 250 iter (turbo)"
    mainClass.set("train.experiments.Bench10MTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV4TinyStoriesPrep") {
    description = "Run CcmcV4TinyStoriesPrep (cefr-kb의 raw.jsonl → data/ccmc-v4-tinystories/{train,val}.bin)"
    mainClass.set("data.CcmcV4TinyStoriesPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV4MergedPrep") {
    description = "v2-pro + v4 9 epochs 통합 prep → data/ccmc-v4-merged/{train,val}.bin"
    mainClass.set("data.CcmcV4MergedPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV5QaPrep") {
    description = "v5_qa Stage 2 dialogues (raw.jsonl) → data/ccmc-v5-qa/{train,val}.bin (v2-pro stage1 BPE 재사용)"
    mainClass.set("data.CcmcV5QaPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV5QaItTrainTurbo") {
    description = "v5-qa-4k IT finetune turbo (3M base v0022). workers=4 (ForkJoinPool common parallelism)"
    mainClass.set("train.experiments.CcmcV5QaItTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf(
        "-Xmx8g",
        "--add-modules=jdk.incubator.vector",
        "-XX:+AlwaysPreTouch",
        "-Djava.util.concurrent.ForkJoinPool.common.parallelism=4",
    )
}

tasks.register<JavaExec>("runCcmcV4MergedSpaceSepPrep") {
    description = "v4-merged 같은 코퍼스에 splitSpaceAsToken=true 새 BPE 학습 → data/ccmc-v4-merged-spacesep/"
    mainClass.set("data.CcmcV4MergedSpaceSepPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV4MergedSpaceSepQuickTrainTurbo") {
    description = "v4-merged-spacesep 1M 모델 2000 iter 빠른 sanity 학습"
    mainClass.set("train.experiments.CcmcV4MergedSpaceSepQuickTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runCcmcV4MergedSpaceSep3MTrainTurbo") {
    description = "v4-merged-spacesep ~3M 모델 (dim 160 · layers 8 · heads 5) 3000 iter 학습"
    mainClass.set("train.experiments.CcmcV4MergedSpaceSep3MTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runSampleV4MergedSpaceSep3M") {
    description = "v0022 best ckpt 재샘플링 (T=0.8, topK=40, topP=0.9, repPenalty=1.15)"
    mainClass.set("sample.SampleV4MergedSpaceSep3MKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runBench5MTurbo") {
    description = "Run Bench5MTurbo — 5M params 500 iter (turbo 백엔드 시간 측정)"
    mainClass.set("train.experiments.Bench5MTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runSamplePromptsFromFile") {
    description = "Run SamplePromptsFromFile (ckpt + 프롬프트 파일로 커스텀 샘플링)"
    mainClass.set("sample.SamplePromptsFromFileKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runChatTurbo") {
    description = "Run ChatTurbo (인터랙티브 대화 REPL — ckpt 인자 필요)"
    mainClass.set("sample.ChatTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
    standardInput = System.`in`
}

tasks.register<JavaExec>("runInferenceApi") {
    description = "Run lightweight inference HTTP API (Turbo 백엔드, ckpt + port 인자)"
    mainClass.set("server.InferenceApiMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runAnalyzeTokens") {
    description = "Run AnalyzeTokensMain (토큰 분포 분석)"
    mainClass.set("data.AnalyzeTokensMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}

tasks.register<JavaExec>("runDebugBPE") {
    description = "Run DebugBPEMain (BPE 디버깅)"
    mainClass.set("data.DebugBPEMainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g", "--add-modules=jdk.incubator.vector", "-XX:+AlwaysPreTouch")
}
