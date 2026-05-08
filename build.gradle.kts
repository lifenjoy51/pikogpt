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
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runBpe") {
    description = "Run BpePrep (BPE 학습 + 인코딩)"
    mainClass.set("data.BpePrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx12g")
}

tasks.register<JavaExec>("runAlphabetPrep") {
    description = "Run AlphabetPrep (알파벳 데이터 준비)"
    mainClass.set("data.AlphabetPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
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
    mainClass.set("train.experiments.TinyHelenTrainKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runTinyHelenTrainVec") {
    description = "Run TinyHelenTrainVec (~1M 파라미터, 벡터 백엔드)"
    mainClass.set("train.experiments.TinyHelenTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runTinyHelenTrainTextbookVec") {
    description = "Run TinyHelenTrainTextbookVec (textbook-only, ~1M 파라미터)"
    mainClass.set("train.experiments.TinyHelenTrainTextbookVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runTinyHelenTrainConversationVec") {
    description = "Run TinyHelenTrainConversationVec (100M conversation, ~1M 파라미터, 12k iter)"
    mainClass.set("train.experiments.TinyHelenTrainConversationVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runConvMixTrainVec") {
    description = "Run ConvMixTrainVec (TinyHelen conv + TinyDialogues age-5 stripped, ~1M 파라미터)"
    mainClass.set("train.experiments.ConvMixTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runConvMixTurnTrainVec") {
    description = "Run ConvMixTurnTrainVec (conv-mix + <|turn|> 토큰, 432k)"
    mainClass.set("train.experiments.ConvMixTurnTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runConvMixTurnNoqTrainVec") {
    description = "Run ConvMixTurnNoqTrainVec (conv-mix-turn 따옴표 제거, 432k)"
    mainClass.set("train.experiments.ConvMixTurnNoqTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runConvMixTurnNoqB128TrainVec") {
    description = "Run ConvMixTurnNoqB128TrainVec (blockSize 128, 436k params)"
    mainClass.set("train.experiments.ConvMixTurnNoqB128TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runConvMixA510TrainVec") {
    description = "Run ConvMixA510TrainVec (TinyHelen + age-5 + age-10, ~18M tok, 432k)"
    mainClass.set("train.experiments.ConvMixA510TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runConvMixA510M773TrainVec") {
    description = "Run ConvMixA510M773TrainVec (a510 데이터 + 773k tied 모델, 12k iter)"
    mainClass.set("train.experiments.ConvMixA510M773TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runConvMixA510M773B128TrainVec") {
    description = "Run ConvMixA510M773B128TrainVec (773k tied + blockSize 128, 8k iter)"
    mainClass.set("train.experiments.ConvMixA510M773B128TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runConvMixCleanA510M773TrainVec") {
    description = "Run ConvMixCleanA510M773TrainVec (clean 데이터 + 773k tied 모델, 12k iter)"
    mainClass.set("train.experiments.ConvMixCleanA510M773TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runConvMixCleanA510M773SwiGLUTrainVec") {
    description = "Run ConvMixCleanA510M773SwiGLUTrainVec (clean + 773k tied + SwiGLU MLP)"
    mainClass.set("train.experiments.ConvMixCleanA510M773SwiGLUTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runConvMixCleanA510M773SwiGLURoPETrainVec") {
    description = "Run ConvMixCleanA510M773SwiGLURoPETrainVec (clean + tied + SwiGLU + RoPE)"
    mainClass.set("train.experiments.ConvMixCleanA510M773SwiGLURoPETrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runDialoguesA510M773SwiGLURoPETrainVec") {
    description = "Run DialoguesA510M773SwiGLURoPETrainVec (TinyDialogues age-5+10 only, tied + SwiGLU + RoPE)"
    mainClass.set("train.experiments.DialoguesA510M773SwiGLURoPETrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runTwoStageBaseTrainVec") {
    description = "Run TwoStageBaseTrainVec (TinyHelen wiki+textbook BASE pretrain — 사실 지식 주입)"
    mainClass.set("train.experiments.TwoStageBaseTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runTwoStageITTrainVec") {
    description = "Run TwoStageITTrainVec (dialogues-a510 IT finetune + 20% BASE replay)"
    mainClass.set("train.experiments.TwoStageITTrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runTwoStageBaseV2TrainVec") {
    description = "Run TwoStageBaseV2TrainVec (TinyHelen wiki+textbook+web+book BASE pretrain, vocab 2000)"
    mainClass.set("train.experiments.TwoStageBaseV2TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runTwoStageITV2TrainVec") {
    description = "Run TwoStageITV2TrainVec (v2 IT finetune + 20% v2 BASE replay)"
    mainClass.set("train.experiments.TwoStageITV2TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runThreeStageDictTrainV4Vec") {
    description = "Run ThreeStageDictTrainV4Vec (Stage 1: dict scratch pretrain, ~2.5k iter, 12 epoch)"
    mainClass.set("train.experiments.ThreeStageDictTrainV4VecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runThreeStageWikiTrainV4Vec") {
    description = "Run ThreeStageWikiTrainV4Vec (Stage 2: wiki finetune from dict ckpt + 30% dict replay)"
    mainClass.set("train.experiments.ThreeStageWikiTrainV4VecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runThreeStageConvTrainV4Vec") {
    description = "Run ThreeStageConvTrainV4Vec (Stage 3: conv finetune from wiki ckpt + multi-replay 15% dict + 15% wiki)"
    mainClass.set("train.experiments.ThreeStageConvTrainV4VecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runThreeStageDictTrainV5Vec") {
    description = "Run ThreeStageDictTrainV5Vec (Stage 1 v5: dict scratch, emb 144 / L9 / H6, ~2.5M params)"
    mainClass.set("train.experiments.ThreeStageDictTrainV5VecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runThreeStageWikiTrainV5Vec") {
    description = "Run ThreeStageWikiTrainV5Vec (Stage 2 v5: wiki finetune from dict ckpt, 30% dict replay)"
    mainClass.set("train.experiments.ThreeStageWikiTrainV5VecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runThreeStageConvTrainV5Vec") {
    description = "Run ThreeStageConvTrainV5Vec (Stage 3 v5: conv finetune from wiki ckpt, 15%+15% multi-replay)"
    mainClass.set("train.experiments.ThreeStageConvTrainV5VecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runCcmcV2ProStage1TrainVec") {
    description = "Run CcmcV2ProStage1TrainVec (CCMC v2-pro Stage 1: binding scratch, 8L×96D×3H SwiGLU+RoPE, vocab 2000)"
    mainClass.set("train.experiments.CcmcV2ProStage1TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runCcmcV2ProStage2TrainVec") {
    description = "Run CcmcV2ProStage2TrainVec (CCMC v2-pro Stage 2: instruction finetune from stage1 ckpt + 25% stage1 replay)"
    mainClass.set("train.experiments.CcmcV2ProStage2TrainVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runEncodeWithExistingMeta") {
    description = "Run EncodeWithExistingMeta (공유 meta.json으로 다른 디렉터리 인코딩)"
    mainClass.set("data.EncodeWithExistingMetaKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx12g")
}

tasks.register<JavaExec>("runSplitByTokenRatio") {
    description = "Run SplitByTokenRatio (record-per-line 입력을 토큰 수 기준으로 train/val 분할)"
    mainClass.set("data.SplitByTokenRatioKt")
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

// turbo 백엔드 — Phase 0~5 진행 중. JDK 21 + Java Vector API 활용.
tasks.register<JavaExec>("runTinyHelenTrainTurbo") {
    description = "Run TinyHelenTrainTurbo (~1M 파라미터, turbo 백엔드)"
    mainClass.set("train.experiments.TinyHelenTrainTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g", "--add-modules=jdk.incubator.vector")
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
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector")
}

tasks.register<JavaExec>("runBench10MVec") {
    description = "Run Bench10MVec — 10M params 250 iter (vec)"
    mainClass.set("train.experiments.Bench10MVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g")
}

tasks.register<JavaExec>("runBench10MTurbo") {
    description = "Run Bench10MTurbo — 10M params 250 iter (turbo)"
    mainClass.set("train.experiments.Bench10MTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx8g", "--add-modules=jdk.incubator.vector")
}

tasks.register<JavaExec>("runCcmcV4TinyStoriesPrep") {
    description = "Run CcmcV4TinyStoriesPrep (cefr-kb의 raw.jsonl → data/ccmc-v4-tinystories/{train,val}.bin)"
    mainClass.set("data.CcmcV4TinyStoriesPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx4g")
}

tasks.register<JavaExec>("runCcmcV4MergedPrep") {
    description = "v2-pro + v4 9 epochs 통합 prep → data/ccmc-v4-merged/{train,val}.bin"
    mainClass.set("data.CcmcV4MergedPrepKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runBench5MVec") {
    description = "Run Bench5MVec — 5M params 500 iter (vec 백엔드 시간 측정)"
    mainClass.set("train.experiments.Bench5MVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g")
}

tasks.register<JavaExec>("runBench5MTurbo") {
    description = "Run Bench5MTurbo — 5M params 500 iter (turbo 백엔드 시간 측정)"
    mainClass.set("train.experiments.Bench5MTurboKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx6g", "--add-modules=jdk.incubator.vector")
}

tasks.register<JavaExec>("runSamplePromptsFromFile") {
    description = "Run SamplePromptsFromFile (ckpt + 프롬프트 파일로 커스텀 샘플링)"
    mainClass.set("sample.SamplePromptsFromFileKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}

tasks.register<JavaExec>("runChatVec") {
    description = "Run ChatVec (인터랙티브 대화 REPL — ckpt 인자 필요)"
    mainClass.set("sample.ChatVecKt")
    classpath = sourceSets.main.get().runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
    standardInput = System.`in`
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
