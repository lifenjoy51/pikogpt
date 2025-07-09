package data

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.IOException
import java.lang.Thread.sleep
import java.net.HttpURLConnection
import java.net.URL
import kotlin.random.Random
import kotlin.system.exitProcess

class StoryGenerator {
    
    private val randomTopics = listOf(
        // 동물 친구들 (Animal Friends)
        "brave little mouse", "friendly dragon", "cute kitten", "rainbow fish", "happy bunny",
        "wise old owl", "dancing bear", "singing bird", "helpful ant", "sleepy turtle",
        "playful monkey", "gentle giant", "kind elephant", "lost puppy", "smart fox",
        "funny penguin", "magic horse", "swimming dolphin", "jumping frog", "busy bee",
        "fluffy sheep", "pretty bird", "funny raccoon", "fast cat", "tall giraffe",
        "little pig", "soft hamster", "baby duck", "colorful parrot", "gentle deer",
        "silly goose", "white swan", "farm cow", "striped zebra", "tiny ant",
        "green frog", "orange cat", "black dog", "pink pig", "red bird",
        "yellow chick", "blue whale", "purple butterfly", "brown bear", "gray wolf",

        // 모험과 탐험 (Adventures and Exploration)
        "treasure hunt", "rocket ship trip", "swimming under water", "climbing big hill",
        "walking in forest", "hot sand place", "pirate ship", "magic time trip",
        "magic castle", "big old animals", "robot friend", "fairy tale trip",
        "super hero help", "dark cave", "flying balloon", "train ride",
        "island adventure", "mountain climbing", "deep sea diving", "space journey",
        "jungle exploration", "desert crossing", "secret tunnel", "hidden door",
        "magic portal", "flying carpet", "lost city", "mystery box",
        "underground world", "sky palace", "magic bridge", "talking map",
        "enchanted forest", "crystal cave", "golden key", "silver castle",

        // 자연과 환경 (Nature and Environment)
        "magic garden", "talking flowers", "magic forest", "rainbow bridge", "snow day fun",
        "sunny day play", "rainy day fun", "yellow leaves", "spring flowers",
        "winter snow", "beach day", "sleeping under stars", "big mountain fire",
        "ocean waves", "water in desert", "green jungle", "cold ice place",
        "beautiful sunset", "morning dew", "butterfly garden", "singing wind",
        "dancing trees", "sparkling river", "peaceful lake", "rolling hills",
        "fluffy clouds", "bright moon", "twinkling stars", "warm sunshine",
        "gentle rain", "cool breeze", "fresh air", "clean water",
        "growing plants", "blooming roses", "fruit trees", "vegetable garden",

        // 가족과 친구 (Family and Friends)
        "best friends", "sharing toys", "helping others", "family picnic", "birthday surprise",
        "grandma visit", "brother sister play", "cousin sleep over", "family car trip",
        "nice neighbor", "playground friends", "new pet", "family game night",
        "cooking with mom", "garden with dad", "reading with grandpa", "making things",
        "baby brother", "big sister", "kind aunt", "funny uncle", "loving family",
        "tea party", "hide and seek", "playing together", "helping mom",
        "walking with dad", "story time", "bedtime hugs", "morning kisses",
        "family dinner", "weekend fun", "holiday joy", "vacation trip",
        "school friends", "park buddies", "neighborhood kids", "classroom mates",

        // 학습과 성장 (Learning and Growth)
        "being brave", "learning new things", "first day school", "fixing problems", "trying hard",
        "making mistakes", "saying sorry", "telling truth", "helping at home", "doing jobs",
        "talking about feelings", "not being scared", "feeling good", "learning to read",
        "counting numbers", "making pictures", "making music", "dancing class",
        "riding bike", "tying shoes", "brushing teeth", "washing hands",
        "eating healthy", "sleeping well", "being kind", "making friends",
        "listening well", "following rules", "being patient", "staying calm",
        "working hard", "never giving up", "believing in yourself", "dreaming big",
        "learning colors", "knowing shapes", "understanding time", "growing tall",

        // 상상력과 창의성 (Imagination and Creativity)
        "magic tree", "flying rug", "talking toys", "pretend friend", "dream trip",
        "wishing well", "magic brush", "singing rocks", "dancing clouds", "color party",
        "changing shapes", "backwards day", "giant tea party", "tiny people",
        "flying house", "toys that sing", "letter game", "number party",
        "invisible friend", "magic wand", "enchanted book", "musical flowers",
        "dancing stars", "laughing sun", "crying moon", "happy rainbow",
        "magic paint", "singing crayons", "dancing pencils", "flying paper",
        "talking pictures", "moving statues", "living dolls", "magic mirror",
        "dream world", "candy land", "toy kingdom", "fairy village",

        // 계절과 휴일 (Seasons and Holidays)
        "Christmas morning", "dress up party", "egg hunt", "summer break",
        "big dinner", "love day cards", "new year party", "birthday wishes",
        "mom day surprise", "dad day gift", "school starts", "picking food",
        "Halloween costume", "thanksgiving feast", "winter wonderland", "spring cleaning",
        "summer vacation", "autumn leaves", "first snow", "melting ice",
        "blooming flowers", "harvest time", "cozy fire", "warm blanket",
        "cold drink", "hot chocolate", "ice cream", "pumpkin pie",
        "gift giving", "party games", "special cake", "fun decorations",

        // 일상 생활 (Daily Life)
        "bedtime story", "morning sun", "making dinner", "cleaning room", "food shopping",
        "doctor visit", "teeth doctor", "book place", "park play", "swimming fun",
        "bike riding", "walking dog", "feeding birds", "watering plants", "building blocks",
        "getting dressed", "eating breakfast", "going to school", "coming home",
        "doing homework", "playing games", "watching TV", "taking bath",
        "brushing hair", "putting on shoes", "packing bag", "making bed",
        "setting table", "washing dishes", "folding clothes", "organizing toys",
        "calling grandma", "visiting friends", "going shopping", "taking walk",

        // 특별한 능력과 마법 (Special Powers and Magic)
        "being invisible", "flying high", "talking to animals", "super strong", "magic stick",
        "reading minds", "changing shape", "stopping time", "making weather", "magic healing",
        "seeing through things", "running fast", "getting big", "getting small", "breathing under water",
        "magic touch", "golden voice", "crystal eyes", "silver wings", "rainbow hair",
        "lightning speed", "gentle power", "kind magic", "helpful spells", "protection charm",
        "luck potion", "happiness dust", "courage pill", "wisdom water", "love energy",
        "peace magic", "healing light", "growing power", "shrinking spell", "floating ability",

        // 직업과 꿈 (Jobs and Dreams)
        "space person", "fire fighter", "doctor helper", "teacher story", "cook making food",
        "person who paints", "music maker", "smart person", "builder", "farm person",
        "book helper", "police helper", "letter bringer", "bread maker", "dancer",
        "animal doctor", "plant grower", "toy maker", "candy maker", "cake decorator",
        "story writer", "picture drawer", "song singer", "movie maker", "game creator",
        "house builder", "bridge maker", "road fixer", "car driver", "bus driver",
        "boat captain", "plane pilot", "train conductor", "bike rider", "walking helper",

        // 음식과 요리 (Food and Cooking)
        "magic cookie", "talking sandwich", "dancing pizza", "singing soup", "happy cake",
        "colorful salad", "sweet apple", "juicy orange", "crunchy carrot", "soft bread",
        "yummy pasta", "tasty rice", "warm milk", "cold juice", "fresh water",
        "chocolate chip", "vanilla ice cream", "strawberry jam", "honey toast", "peanut butter",
        "cheese sandwich", "banana split", "fruit salad", "vegetable soup", "chicken dinner",

        // 장난감과 게임 (Toys and Games)
        "magic blocks", "talking doll", "flying kite", "bouncing ball", "rolling car",
        "building castle", "puzzle pieces", "coloring book", "drawing paper", "clay animals",
        "puppet show", "dress up box", "musical instruments", "dancing robot", "singing teddy",
        "racing cars", "flying planes", "sailing boats", "climbing jungle gym", "swinging high",
        "sliding down", "jumping rope", "playing tag", "hide and seek", "follow leader",

        // 색깔과 모양 (Colors and Shapes)
        "red balloon", "blue sky", "green grass", "yellow sun", "purple flower",
        "orange pumpkin", "pink dress", "white snow", "black cat", "brown dog",
        "round ball", "square box", "triangle hat", "circle wheel", "rectangle door",
        "star shape", "heart love", "diamond ring", "oval egg", "spiral shell",
        "striped shirt", "spotted dog", "checkered flag", "polka dot dress", "rainbow colors",

        // 교통수단 (Transportation)
        "magic bus", "flying car", "talking train", "sailing boat", "riding horse",
        "big truck", "fast motorcycle", "slow turtle", "rolling wheel", "floating balloon",
        "subway ride", "taxi trip", "walking feet", "running shoes", "climbing stairs",
        "elevator ride", "escalator fun", "bridge crossing", "tunnel journey", "road trip"
    )
    
    fun getRandomTopic(): String {
        return randomTopics[Random.nextInt(randomTopics.size)]
    }

    fun generateStory(prompt: String): String {
        val enhancedPrompt = createEnhancedPrompt(prompt)
        
        return try {
            val startAt = System.currentTimeMillis()

            // JSON 요청 본문 생성 (LM Studio 호환)
            val requestBody = """
                {
                    "model": "google/gemma-3-1b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "${enhancedPrompt.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")}"
                        }
                    ],
                    "temperature": 0.7
                }
            """.trimIndent()

            // URLConnection을 사용한 HTTP 요청 (LM Studio 엔드포인트)
            println("   [HTTP 요청 전송 시작] URL: http://127.0.0.1:1234/v1/chat/completions")

            val url = URL("http://127.0.0.1:1234/v1/chat/completions")
            val connection = url.openConnection() as HttpURLConnection

            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.doOutput = true
            connection.connectTimeout = 30000   // 30초
            connection.readTimeout = 300000     // 5분

            // 요청 바디 전송
            connection.outputStream.use { os ->
                os.write(requestBody.toByteArray())
            }

            val responseCode = connection.responseCode
            println("   [응답 수신 완료] 상태코드: $responseCode")

            val responseBody = if (responseCode == 200) {
                connection.inputStream.bufferedReader().use { it.readText() }
            } else {
                val errorText = connection.errorStream?.bufferedReader()?.use { it.readText() } ?: "응답 없음"
                throw IOException("HTTP 오류: $responseCode\n응답: $errorText")
            }

            println("   [응답 바디 크기] ${responseBody.length}바이트")

            // kotlinx.serialization을 사용한 JSON 파싱
            val jsonResponse = Json.parseToJsonElement(responseBody).jsonObject
            val choices = jsonResponse["choices"]?.jsonArray
            if (choices.isNullOrEmpty()) {
                throw IOException("응답에서 'choices'를 찾을 수 없습니다: $responseBody")
            }

            val message = choices[0].jsonObject["message"]?.jsonObject
            if (message == null) {
                throw IOException("응답에서 'message'를 찾을 수 없습니다: $responseBody")
            }

            var result = message["content"]?.jsonPrimitive?.content
                ?: throw IOException("응답에서 'content'를 찾을 수 없습니다: $responseBody")

            // <think> 태그와 그 내용을 제거 (DeepSeek 모델의 사고 과정 제거)
            val thinkStartPattern = "<think>"
            val thinkEndPattern = "</think>"

            while (result.contains(thinkStartPattern)) {
                val startIndex = result.indexOf(thinkStartPattern)
                val endIndex = result.indexOf(thinkEndPattern, startIndex)

                result = if (endIndex != -1) {
                    // <think>부터 </think>까지 제거
                    result.substring(0, startIndex) +
                        result.substring(endIndex + thinkEndPattern.length)
                } else {
                    // </think>가 없으면 <think>부터 끝까지 제거
                    result.substring(0, startIndex)
                }
            }

            // 추가 정리: 남은 공백이나 불필요한 문자 제거
            result = result.trim()

            println("   [응답 완료] elapsed=${(System.currentTimeMillis() - startAt) / 1000}s")
            // sleep(6_000) // 6초 대기 (API 제한 회피를 위한 간단한 방법)
            result

        } catch (e: IOException) {
            println("   [오류 상세] IOException 발생: ${e.message}")
            throw IOException("HTTP API 호출 중 오류 발생: ${e.message}", e)
        } catch (e: Exception) {
            println("   [오류 상세] 예상치 못한 오류: ${e.javaClass.simpleName} - ${e.message}")
            throw IOException("예상치 못한 오류 발생: ${e.message}", e)
        }
    }

    fun validateStoryQuality(story: String): Boolean {
        val validationPrompt = createValidationPrompt(story)

        return try {
            val requestBody = """
                {
                    "model": "google/gemma-3-1b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "${validationPrompt.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")}"
                        }
                    ],
                    "temperature": 0.3
                }
            """.trimIndent()

            val url = URL("http://127.0.0.1:1234/v1/chat/completions")
            val connection = url.openConnection() as HttpURLConnection

            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.doOutput = true
            connection.connectTimeout = 30000
            connection.readTimeout = 300000

            connection.outputStream.use { os ->
                os.write(requestBody.toByteArray())
            }

            val responseCode = connection.responseCode
            if (responseCode != 200) {
                println("   [검증 실패] HTTP 오류: $responseCode")
                return false
            }

            val responseBody = connection.inputStream.bufferedReader().use { it.readText() }

            // kotlinx.serialization을 사용한 JSON 파싱
            val jsonResponse = Json.parseToJsonElement(responseBody).jsonObject
            val choices = jsonResponse["choices"]?.jsonArray
            if (choices.isNullOrEmpty()) {
                println("   [검증 실패] 응답에서 'choices'를 찾을 수 없습니다")
                return false
            }

            val message = choices[0].jsonObject["message"]?.jsonObject
            if (message == null) {
                println("   [검증 실패] 응답에서 'message'를 찾을 수 없습니다")
                return false
            }

            var result = message["content"]?.jsonPrimitive?.content ?: ""
            if (result.isNullOrBlank()) {
                println("   [검증 실패] 응답에서 'content'를 찾을 수 없습니다")
                return false
            }

            // <think> 태그와 그 내용을 제거 (DeepSeek 모델의 사고 과정 제거)
            val thinkStartPattern = "<think>"
            val thinkEndPattern = "</think>"

            while (result.contains(thinkStartPattern)) {
                val startIndex = result.indexOf(thinkStartPattern)
                val endIndex = result.indexOf(thinkEndPattern, startIndex)

                result = if (endIndex != -1) {
                    result.substring(0, startIndex) +
                        result.substring(endIndex + thinkEndPattern.length)
                } else {
                    result.substring(0, startIndex)
                }
            }

            result = result.trim()

            val isValid = result.lowercase().contains("yes") || result.lowercase().contains("적합")
            println("   [검증 결과] ${if (isValid) "통과" else "실패"}: $result")

            return isValid

        } catch (e: Exception) {
            println("   [검증 오류] ${e.message}")
            return false
        }
    }

    private fun createValidationPrompt(story: String): String {
        return """
Please evaluate this story:

"$story"

Does it satisfy ALL 6 conditions? Answer YES or NO:

1. Is it written with simple words and sentences that a 6-year-old child can understand?
2. Does it have a clear story structure (beginning-problem-solution-ending)?
3. Is the story completely finished? (Not ending abruptly in the middle?)
4. Does it avoid proper nouns (names of people, places, brands like Alex, Benny, Charlie, America, Texas, Lego)?
5. Does it avoid non-standard sounds and interjections (like aaaaaaaaaaaaaa, ahhh, baaah, choochooh, heeeey, lalalalaaaa, soooooo, weeeeeee, bzzzt, grrr, wump)?
6. Does it avoid made-up compound words, typos, technical terms, and unclear single letters (like aaaabook, allbuilder, betterbuilder, accidently, adventurousy, proboscis, chrysalis, b, c, d, g, m, p, q, r, s, t, x)?

If ALL conditions are met, answer "YES". If ANY condition is not met, answer "NO" and briefly explain why.

Answer:
        """.trimIndent()
    }

    private fun createEnhancedPrompt(userPrompt: String): String {
        
        return """
Write a COMPLETE story about $userPrompt for a 6-year-old child. CRITICAL: The story MUST have a clear beginning, middle, and proper ending. The story MUST be completely finished with a satisfying conclusion. Do not stop in the middle.

IMPORTANT: Return ONLY the story text as a single continuous line without any line breaks, paragraphs, titles, chapter headings, explanations, or introductory text. Do not include phrases like "Here is a story" or "Chapter 1" or any formatting. Start immediately with the story content and write everything as one long sentence or connected sentences without any line breaks or paragraph breaks.

Requirements:
- Use only basic words that a 6-year-old can understand
- Simple sentence structure
- shorter but COMPLETE
- Include characters, problem, and HAPPY ENDING
- Write everything in one continuous line of text
- MUST finish the story completely - do not stop in the middle
- End with phrases like "and they lived happily ever after" or "and everyone was happy" or "the end"

AVOID THESE COMPLETELY:
1. Proper nouns (names of people, places, brands): NO Alex, Benny, Charlie, America, Texas, Lego, etc.
2. Sound effects and interjections: NO aaaaaaaaaaaaaa, ahhh, baaah, choochooh, heeeey, lalalalaaaa, soooooo, weeeeeee, bzzzt, grrr, wump
3. Made-up compound words: NO aaaabook, allbuilder, betterbuilder, birdfriend, burritoseos, dayeos, friendseos, happyeos, homeeos
4. Typos or wrong spellings: NO accidently, adventurousy, carefuly, ssorry, ththank
5. Hard or technical words: NO proboscis, chrysalis, galleon, acacia, alas, centimeters
6. Single letters or unclear words: NO b, c, d, g, m, p, q, r, s, t, x

Use only simple, common words. Call characters "the boy", "the girl", "the cat", "the dog", etc. Use "said" not sound effects.

Example format: Once upon a time there was a little rabbit who lived in a forest and one day he found a magic carrot that could talk and the carrot said hello little rabbit I need your help to find my way back to the garden and the rabbit said yes I will help you and together they walked through the forest until they found the beautiful garden and the carrot was so happy to be home and they became best friends forever and lived happily ever after.

Write the entire COMPLETE story as one line with a proper ending:
        """.trimIndent()
    }
}

fun main() {
    // 설정 변수들
    val targetTotalSize = 1_000_000_000  // 최종 결과물 목표 크기 (바이트)
    val outputFile = "stories-${System.currentTimeMillis()}.txt"

    println("=== 랜덤 이야기 생성기 (LM Studio) ===")
    println("LM Studio 엔드포인트: http://localhost:1234/v1/chat/completions")
    println("사용 모델: google/gemma-3-1b")
    println("목표 총 크기: ${targetTotalSize}바이트")
    println("출력 파일: $outputFile")
    println("목표 크기 도달 시까지 계속 생성...")
    println()
    
    try {
        val generator = StoryGenerator()
        val outputFileHandle = File(outputFile)
        
        // 기존 파일 삭제 (새로 시작)
        if (outputFileHandle.exists()) {
            outputFileHandle.delete()
        }
        
        var currentSize = 0
        var storyIndex = 0

        while (currentSize < targetTotalSize) {
            storyIndex++
            val topic = generator.getRandomTopic()
            println("\n ${storyIndex}. 생성 중... 주제: $topic")
            
            try {
                // 기본 이야기 생성
                val story = generator.generateStory(topic)

                // 이야기 품질 검증
                println("   [품질 검증] 6세 아동 적합성, 기승전결, 완성도 확인...")
                val isQualityValid = generator.validateStoryQuality(story)

                if (!isQualityValid) {
                    println("   [품질 검증 실패] 조건을 만족하지 않아 이야기를 건너뜁니다.")
                    storyIndex-- // 인덱스 되돌리기
                    sleep(1000)
                    continue
                }
                
                // 구분자 포함한 예상 크기 계산
                val storyWithSeparator = if (storyIndex == 1) story else "\n<|eos|>\n$story"
                val estimatedNewSize = currentSize + storyWithSeparator.toByteArray().size
                
                // 목표 크기를 넘으면 생성 중단
                if (estimatedNewSize > targetTotalSize) {
                    println("   목표 크기 도달! 생성 중단.")
                    break
                }

                // 파일에 즉시 기록 (검증을 통과한 이야기만)
                outputFileHandle.appendText("$story <|eos|>\n")
                
                currentSize = estimatedNewSize
                println("   완료! (${story.toByteArray().size}바이트) - 현재 총 크기: ${currentSize}바이트")
                
            } catch (e: Exception) {
                println("   [생성 실패] 주제 '$topic' 이야기 생성 중 오류 발생: ${e.message}")
                println("   [재시도] 다른 주제로 재시도합니다...")
                storyIndex-- // 인덱스 되돌리기
                
                // 너무 많은 연속 실패를 방지하기 위한 간단한 대기
                sleep(1000)
                continue
            }
        }
        
        val actualSize = outputFileHandle.length()
        println("\n=== 생성 완료! ===")
        println("총 이야기 개수: $storyIndex")
        println("목표 크기: ${targetTotalSize}바이트")
        println("실제 크기: ${actualSize}바이트")
        println("저장 위치: $outputFile")
        
    } catch (e: Exception) {
        println("오류: ${e.message}")
        exitProcess(1)
    }
}