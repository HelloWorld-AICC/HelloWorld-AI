import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.*;

public class TranslatorTest {

    private final WebClient webClient;
    private final String apiKey;

    public TranslatorTest() {
        this.apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("OPENAI_API_KEY not set");
        }
        this.webClient = WebClient.builder()
                .baseUrl("https://api.openai.com/v1/chat/completions")
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    // ---------- 네 기존 시그니처 유지 ----------
    private Mono<String> translateToUserLanguage(String text, String targetLanguage) {
        // system 프롬프트: 형식/볼드/JSON 규칙
        var systemMessage = Map.of("role", "system", "content", String.join("\n",
            "You are a precise translator from Korean to the target language.",
            "Rules:",
            "1) Translate the SOURCE faithfully and naturally into the target language.",
            "2) Preserve any existing Markdown formatting, especially **bold**; if none exists, add **bold** only to the most important legal terms, actions, deadlines, and institutions (use sparingly).",
            "3) Output MUST be JSON with a single key \"answer\" whose value is the translation string with Markdown intact.",
            "4) No explanations, no code fences, no extra keys. JSON only.",
            "5) Behave deterministically (temperature 0 behavior)."
        ));

        // user 프롬프트: 최소 few-shot(영어) 1개 + 실제 입력
        String userContent =
            "TARGET: " + targetLanguage + "\n" +
            "FORMAT: Return JSON only with a single key \\\"answer\\\".\n" +
            "Example (Korean -> English):\n" +
            "SOURCE:\n```SOURCE\n관할 노동청에 신고하면 고용주에게 법적 의무를 이행하도록 요구할 수 있습니다.\n```\n" +
            "OUTPUT:\n" +
            "{\"answer\":\"By **reporting to the local Labor Office**, you can require the employer to **fulfill legal obligations**.\"}\n\n" +
            "Now translate the following SOURCE:\n" +
            "```SOURCE\n" + text + "\n```";

        var userMessage = Map.of("role", "user", "content", userContent);

        Map<String, Object> body = new HashMap<>();
        body.put("model", "gpt-4o-mini");     // ← 모델만 업데이트
        body.put("temperature", 0);
        body.put("messages", List.of(systemMessage, userMessage));
        // body.put("response_format", Map.of("type", "json_object")); // (옵션) 지원되면 켜기

        return webClient.post()
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .map(res -> {
                    var choices = (List<Map<String, Object>>) res.get("choices");
                    var msg = (Map<String, Object>) choices.get(0).get("message");
                    var content = (String) msg.get("content"); // e.g., {"answer":"..."}
                    try {
                        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                        Map<String, Object> json = mapper.readValue(content, Map.class);
                        return (String) json.getOrDefault("answer", "");
                    } catch (Exception e) {
                        // 혹시 모델이 JSON 외 형식으로 답하면 그대로 반환
                        return content;
                    }
                });
    }

    // ---------- 실행용 main ----------
    public static void main(String[] args) {
        // 네가 준 한국어 예시(JSON)에서 answer 본문만 넣어 테스트해도 되고,
        // 아래처럼 간단 문장으로 먼저 확인해도 됨.
        String koreanSample =
            "불법체류 상태에서 공장에서 근무하다가 다쳤고, 회사가 병원비를 지급할 수 없다고 하는 경우, " +
            "다음과 같은 해결 방안을 고려해보실 수 있습니다. **산업재해보상보험 청구**, 관할 노동청 신고, 법률 상담, 의료 지원 요청 등.";

        var client = new TranslatorTest();
        String target = "English"; // 영어로 테스트

        String result = client.translateToUserLanguage(koreanSample, target).block();
        System.out.println("=== TRANSLATION ANSWER (should contain **bold**) ===");
        System.out.println(result);
        System.out.println("Has bold? " + (result != null && result.contains("**")));
    }
}
