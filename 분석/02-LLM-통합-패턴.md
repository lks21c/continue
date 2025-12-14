# LLM 통합 패턴

> **목적**: Continue의 LLM 추상화 레이어, 프로바이더 구현, 스트리밍 패턴을 분석하여 hdsp-agent의 LLM 통합 설계에 활용

---

## 1. 개요

Continue는 76개 이상의 LLM 프로바이더를 지원하기 위해 추상화 레이어를 사용합니다. 핵심 클래스인 `BaseLLM`은 모든 프로바이더가 구현해야 하는 인터페이스를 정의합니다.

### 핵심 파일
- `core/llm/index.ts` (43KB) - BaseLLM 추상 클래스
- `core/llm/llms/*.ts` - 76+ 프로바이더 구현
- `core/llm/openaiTypeConverters.ts` (27KB) - 메시지 타입 변환
- `core/llm/streamChat.ts` (8KB) - 스트리밍 오케스트레이션

---

## 2. ILLM 인터페이스

모든 LLM 프로바이더가 구현해야 하는 핵심 인터페이스입니다.

```typescript
// core/index.d.ts
export interface ILLM {
  // 프로바이더 식별
  get providerName(): string;
  get underlyingProviderName(): string;

  // 기본 속성
  model: string;
  uniqueId: string;
  contextLength: number;
  completionOptions: CompletionOptions;

  // 핵심 메서드 (Async Generator 패턴)
  streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options?: LLMFullCompletionOptions,
  ): AsyncGenerator<ChatMessage, PromptLog>;

  streamComplete(
    prompt: string,
    signal: AbortSignal,
    options?: LLMFullCompletionOptions,
  ): AsyncGenerator<string, PromptLog>;

  streamFim(
    prefix: string,
    suffix: string,
    signal: AbortSignal,
  ): AsyncGenerator<string, PromptLog>;

  // 동기 메서드
  chat(messages: ChatMessage[], signal: AbortSignal): Promise<ChatMessage>;
  complete(prompt: string, signal: AbortSignal): Promise<string>;

  // 유틸리티
  countTokens(text: string): number;
  embed(chunks: string[]): Promise<number[][]>;
  rerank(query: string, chunks: Chunk[]): Promise<number[]>;

  // 기능 감지
  supportsImages(): boolean;
  supportsCompletions(): boolean;
  supportsPrefill(): boolean;
  supportsFim(): boolean;
}
```

---

## 3. BaseLLM 추상 클래스

모든 프로바이더의 기반이 되는 추상 클래스입니다.

```typescript
// core/llm/index.ts
export abstract class BaseLLM implements ILLM {
  // 정적 프로퍼티 (서브클래스에서 오버라이드)
  static providerName: string;
  static defaultOptions: Partial<LLMOptions> | undefined = undefined;

  // Provider capabilities (서브클래스에서 오버라이드)
  protected supportsReasoningField: boolean = false;
  protected supportsReasoningDetailsField: boolean = false;

  // 인스턴스 속성
  uniqueId: string;
  model: string;
  title?: string;
  apiKey?: string;
  apiBase?: string;
  _contextLength: number | undefined;
  completionOptions: CompletionOptions;
  requestOptions?: RequestOptions;

  // 프로바이더 이름 getter
  get providerName(): string {
    return (this.constructor as typeof BaseLLM).providerName;
  }

  // 기능 감지 메서드
  supportsImages(): boolean {
    return modelSupportsImages(
      this.providerName,
      this.model,
      this.title,
      this.capabilities,
    );
  }

  supportsCompletions(): boolean {
    if (["groq", "mistral", "deepseek"].includes(this.providerName)) {
      return false;
    }
    return true;
  }

  supportsPrefill(): boolean {
    return ["ollama", "anthropic", "mistral"].includes(this.providerName);
  }
}
```

---

## 4. 프로바이더 구현 예시

### 4.1 OpenAI 프로바이더

```typescript
// core/llm/llms/OpenAI.ts (20KB)
class OpenAI extends BaseLLM {
  static providerName = "openai";
  static defaultOptions: Partial<LLMOptions> = {
    model: "gpt-4o",
    contextLength: 128_000,
    maxTokens: 4096,
  };

  // Reasoning 지원 (o1, o3 모델)
  protected supportsReasoningField: boolean = true;

  // 스트리밍 채팅 구현
  async *streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options?: LLMFullCompletionOptions,
  ): AsyncGenerator<ChatMessage, PromptLog> {
    // 1. 요청 바디 생성
    const body = toChatBody(messages, {
      ...this.completionOptions,
      ...options,
    });

    // 2. API 호출
    const response = await this.fetch(`${this.apiBase}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ ...body, stream: true }),
      signal,
    });

    // 3. SSE 스트리밍 처리
    for await (const chunk of streamSse(response)) {
      const message = fromChatCompletionChunk(chunk);
      yield message;
    }

    // 4. PromptLog 반환
    return {
      modelTitle: this.title,
      completionOptions: options,
      prompt: JSON.stringify(messages),
      completion: "", // 청크에서 누적
    };
  }
}
```

### 4.2 Anthropic 프로바이더

```typescript
// core/llm/llms/Anthropic.ts (13KB)
class Anthropic extends BaseLLM {
  static providerName = "anthropic";
  static defaultOptions: Partial<LLMOptions> = {
    model: "claude-sonnet-4-20250514",
    contextLength: 200_000,
  };

  // Anthropic 전용 도구 변환
  private convertToolToAnthropicTool(tool: Tool): AnthropicTool {
    return {
      name: tool.function.name,
      description: tool.function.description,
      input_schema: tool.function.parameters,
    };
  }

  // Anthropic API 파라미터 변환
  public convertArgs(options: CompletionOptions): MessageCreateParams {
    return {
      model: options.model,
      max_tokens: options.maxTokens ?? 4096,
      tools: options.tools?.map(t => this.convertToolToAnthropicTool(t)),
      // Thinking 모드 (Claude 3.5+)
      thinking: options.reasoning
        ? {
            type: "enabled",
            budget_tokens: options.reasoningBudgetTokens ?? 10000,
          }
        : undefined,
      stream: true,
    };
  }
}
```

### 4.3 Ollama 프로바이더

```typescript
// core/llm/llms/Ollama.ts (22KB)
class Ollama extends BaseLLM {
  static providerName = "ollama";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "http://localhost:11434",
  };

  // FIM (Fill-in-Middle) 지원
  supportsFim(): boolean {
    return true;
  }

  // 로컬 모델 목록 조회
  async listModels(): Promise<string[]> {
    const response = await this.fetch(`${this.apiBase}/api/tags`);
    const data = await response.json();
    return data.models.map((m: any) => m.name);
  }

  // Ollama 전용 스트리밍
  async *streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options?: LLMFullCompletionOptions,
  ): AsyncGenerator<ChatMessage, PromptLog> {
    const response = await this.fetch(`${this.apiBase}/api/chat`, {
      method: "POST",
      body: JSON.stringify({
        model: this.model,
        messages: this.convertMessages(messages),
        stream: true,
        options: {
          num_ctx: this.contextLength,
          temperature: options?.temperature,
        },
      }),
      signal,
    });

    // NDJSON 스트리밍
    for await (const line of streamLines(response)) {
      const json = JSON.parse(line);
      if (json.message?.content) {
        yield {
          role: "assistant",
          content: json.message.content,
        };
      }
    }

    return this.createPromptLog(messages, options);
  }
}
```

---

## 5. 메시지 타입 변환

다양한 LLM API 간의 메시지 형식을 변환합니다.

### 5.1 ChatMessage 타입

```typescript
// core/index.d.ts
export type ChatMessageRole = "user" | "assistant" | "thinking" | "system" | "tool";

export interface ChatMessage {
  role: ChatMessageRole;
  content: MessageContent;
  toolCalls?: ToolCallDelta[];
  usage?: Usage;
  metadata?: Record<string, unknown>;
}

export type MessageContent = string | MessagePart[];

export type MessagePart =
  | { type: "text"; text: string }
  | { type: "imageUrl"; imageUrl: { url: string } };
```

### 5.2 OpenAI 타입 변환기

```typescript
// core/llm/openaiTypeConverters.ts
export function toChatMessage(
  message: ChatMessage,
  options: CompletionOptions,
  prevMessage?: ChatMessage,
): ChatCompletionMessageParam | null {
  // Thinking 메시지 병합 (Claude용)
  if (message.role === "thinking" && prevMessage?.role === "assistant") {
    return null; // 이전 메시지에 병합됨
  }

  const msg: ChatCompletionMessageParam = {
    role: message.role === "thinking" ? "assistant" : message.role,
    content: typeof message.content === "string"
      ? message.content
      : message.content.map(part => ({
          type: part.type,
          ...(part.type === "text" ? { text: part.text } : { image_url: part.imageUrl }),
        })),
  };

  // Tool Calls 변환
  if (message.toolCalls?.length) {
    msg.tool_calls = message.toolCalls.map(tc => ({
      id: tc.id!,
      type: "function",
      function: {
        name: tc.function!.name!,
        arguments: tc.function!.arguments!,
      },
    }));
  }

  return msg;
}

// 스트리밍 청크 → ChatMessage
export function fromChatCompletionChunk(
  chunk: ChatCompletionChunk
): ChatMessage {
  const delta = chunk.choices[0]?.delta;

  return {
    role: delta?.role ?? "assistant",
    content: delta?.content ?? "",
    toolCalls: delta?.tool_calls?.map(tc => ({
      id: tc.id,
      type: tc.type,
      function: {
        name: tc.function?.name,
        arguments: tc.function?.arguments,
      },
    })),
  };
}
```

---

## 6. 스트리밍 패턴

### 6.1 Async Generator 기반 스트리밍

```typescript
// core/llm/streamChat.ts
export async function* llmStreamChat(
  configHandler: ConfigHandler,
  abortController: AbortController,
  msg: StreamChatMessage,
  ide: IDE,
): AsyncGenerator<ChatMessage, PromptLog> {
  // 1. 선택된 모델 가져오기
  const config = await configHandler.getConfig();
  const model = config.selectedModelByRole.chat;

  // 2. 스트리밍 시작
  const generator = model.streamChat(
    msg.messages,
    abortController.signal,
    msg.completionOptions,
  );

  // 3. 청크 전달
  for await (const chunk of generator) {
    // 도구 호출 감지
    if (chunk.toolCalls?.length) {
      yield* handleToolCalls(chunk.toolCalls, model, ide);
    }

    yield chunk;
  }

  // 4. 최종 PromptLog 반환
  return generator.return();
}
```

### 6.2 SSE 스트림 파싱

```typescript
// core/util/stream.ts
export async function* streamSse(
  response: Response
): AsyncGenerator<any> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE 이벤트 파싱
    const lines = buffer.split("\n");
    buffer = lines.pop()!;

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;

        try {
          yield JSON.parse(data);
        } catch {
          // 파싱 실패 무시
        }
      }
    }
  }
}
```

---

## 7. 토큰 카운팅

### 7.1 인코딩 선택

```typescript
// core/llm/countTokens.ts
import { get_encoding, Tiktoken } from "js-tiktoken";
import { llamaTokenizer } from "../vendor/llama-tokenizer.js";

function encodingForModel(modelName: string): Tiktoken | typeof llamaTokenizer {
  if (modelName.includes("llama") || modelName.includes("mistral")) {
    return llamaTokenizer;
  }
  return get_encoding("cl100k_base");
}

export function countTokens(text: string, modelName: string): number {
  const encoding = encodingForModel(modelName);
  return encoding.encode(text).length;
}
```

### 7.2 메시지 컴파일 및 프루닝

```typescript
// core/llm/countTokens.ts
export function pruneRawPromptFromTop(
  prompt: string,
  maxLength: number,
  llm: BaseLLM,
): string {
  const tokens = llm.countTokens(prompt);
  if (tokens <= maxLength) return prompt;

  // 상단에서 토큰 제거
  const lines = prompt.split("\n");
  while (llm.countTokens(lines.join("\n")) > maxLength && lines.length > 1) {
    lines.shift();
  }

  return lines.join("\n");
}
```

---

## 8. 팩토리 패턴

### 8.1 프로바이더 등록

```typescript
// core/llm/llms/index.ts
export const LLMClasses: (typeof BaseLLM)[] = [
  Anthropic,
  OpenAI,
  Ollama,
  Azure,
  Bedrock,
  Cohere,
  Gemini,
  Groq,
  Mistral,
  // ... 76+ 프로바이더
];

const LLM_PROVIDER_MAP: Record<string, typeof BaseLLM> = {};
for (const cls of LLMClasses) {
  LLM_PROVIDER_MAP[cls.providerName] = cls;
}
```

### 8.2 동적 인스턴스 생성

```typescript
// core/llm/llms/index.ts
export function llmFromDescription(
  desc: ModelDescription,
  requestOptions?: RequestOptions,
): BaseLLM {
  const ProviderClass = LLM_PROVIDER_MAP[desc.provider];

  if (!ProviderClass) {
    throw new Error(`Unknown LLM provider: ${desc.provider}`);
  }

  return new ProviderClass({
    model: desc.model,
    apiKey: desc.apiKey,
    apiBase: desc.apiBase,
    contextLength: desc.contextLength,
    completionOptions: desc.completionOptions ?? {},
    requestOptions,
  });
}
```

---

## 9. hdsp-agent 적용 방안

### 9.1 권장 아키텍처

```typescript
// hdsp-agent/llm/base.ts
export interface ILLMProvider {
  providerName: string;
  model: string;
  contextLength: number;

  // Async Generator 패턴 유지
  streamChat(
    messages: Message[],
    signal: AbortSignal,
    options?: CompletionOptions,
  ): AsyncGenerator<Message>;

  // 유틸리티
  countTokens(text: string): number;
}

// hdsp-agent/llm/ollama.ts
export class OllamaProvider implements ILLMProvider {
  providerName = "ollama";

  constructor(
    public model: string,
    public apiBase = "http://localhost:11434",
  ) {}

  async *streamChat(
    messages: Message[],
    signal: AbortSignal,
  ): AsyncGenerator<Message> {
    const response = await fetch(`${this.apiBase}/api/chat`, {
      method: "POST",
      body: JSON.stringify({
        model: this.model,
        messages,
        stream: true,
      }),
      signal,
    });

    for await (const line of streamLines(response)) {
      const json = JSON.parse(line);
      yield { role: "assistant", content: json.message?.content ?? "" };
    }
  }
}
```

### 9.2 Jupyter 통합 고려사항

```python
# hdsp-agent/llm/provider.py (Python 버전)
from abc import ABC, abstractmethod
from typing import AsyncGenerator
import aiohttp

class BaseLLMProvider(ABC):
    """LLM 프로바이더 추상 클래스"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
    ) -> AsyncGenerator[dict, None]:
        pass

class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(self, model: str, api_base: str = "http://localhost:11434"):
        self.model = model
        self.api_base = api_base

    async def stream_chat(
        self,
        messages: list[dict],
    ) -> AsyncGenerator[dict, None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
            ) as response:
                async for line in response.content:
                    data = json.loads(line)
                    yield {"role": "assistant", "content": data.get("message", {}).get("content", "")}
```

### 9.3 구현 체크리스트

- [ ] `ILLMProvider` 인터페이스 정의
- [ ] Async Generator 기반 스트리밍 구현
- [ ] Ollama 프로바이더 구현 (로컬 개발용)
- [ ] OpenAI 호환 프로바이더 구현
- [ ] 토큰 카운팅 유틸리티
- [ ] 팩토리 함수 구현
- [ ] Jupyter Comm 스트리밍 연동

---

## 10. 참고 파일

| 파일 | 크기 | 용도 |
|------|------|------|
| `core/llm/index.ts` | 43KB | BaseLLM 추상 클래스 |
| `core/llm/llms/OpenAI.ts` | 20KB | OpenAI 프로바이더 |
| `core/llm/llms/Anthropic.ts` | 13KB | Anthropic 프로바이더 |
| `core/llm/llms/Ollama.ts` | 22KB | Ollama 프로바이더 |
| `core/llm/openaiTypeConverters.ts` | 27KB | 타입 변환 |
| `core/llm/countTokens.ts` | 17KB | 토큰 카운팅 |
| `core/llm/streamChat.ts` | 8KB | 스트리밍 오케스트레이션 |

---

*이전 문서: [01-프로젝트-개요.md](./01-프로젝트-개요.md)*
*다음 문서: [03-프로토콜-통신.md](./03-프로토콜-통신.md)*
