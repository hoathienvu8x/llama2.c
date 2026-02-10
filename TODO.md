Để hoàn thiện hàm `main` cho một chatbot từ `llama2.c`, chúng ta cần sửa
đổi logic mặc định của **Karpathy**: thay vì nhận một prompt từ tham số
dòng lệnh rồi thoát, chúng ta sẽ đưa nó vào một vòng lặp `while` và duy
trì trạng thái của `pos` (vị trí token) để mô hình có "trí nhớ" trong cuộc
hội thoại.

Dưới đây là mã nguồn hoàn chỉnh cho hàm `main`, được thiết kế để tích hợp
trực tiếp vào file `run.c`.

**Mã nguồn hàm `main` cho Chatbot C thuần**

```c
int main(int argc, char *argv[]) {
    // --- 1. Khởi tạo tham số mặc định ---
    char *checkpoint_path = NULL;  // File .bin của model
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.9f;      // Độ sáng tạo
    float topp = 0.9f;             // Top-p sampling
    int steps = 256;               // Số token tối đa mỗi lần trả lời
    char *prompt = NULL;           // Không dùng prompt cố định
    unsigned long long rng_seed = 0;

    // Đọc checkpoint từ tham số dòng lệnh
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <checkpoint_path> [temperature] [topp] [steps]\n", argv[0]);
        return 1;
    }
    checkpoint_path = argv[1];
    if (argc >= 3) temperature = atof(argv[2]);
    if (argc >= 4) topp = atof(argv[3]);
    if (argc >= 5) steps = atof(argv[4]);

    // Khởi tạo RNG
    if (rng_seed == 0) rng_seed = (unsigned int)time(NULL);

    // --- 2. Load Model và Tokenizer vào RAM ---
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // --- 3. Vòng lặp Chat (REPL) ---
    RunState state;
    build_run_state(&state, &transformer);

    printf("--- Llama2.c Chatbot (Phien ban C thuan) ---\n");
    printf("Go 'exit' de thoat, 'reset' de xoa bo nho tam.\n");

    int next;        // Token tiếp theo được dự đoán
    int token;       // Token hiện tại
    int pos = 0;     // Vi trí hiện tại trong chuỗi ngữ cảnh (Context Window)
    char user_input[2048];

    while (pos < transformer.config.seq_len) {
        printf("\nUSER: ");
        if (fgets(user_input, sizeof(user_input), stdin) == NULL) break;
        
        // Xử lý lệnh đặc biệt
        if (strncmp(user_input, "exit", 4) == 0) break;
        if (strncmp(user_input, "reset", 5) == 0) {
            pos = 0;
            printf("--- Da reset ngu canh ---\n");
            continue;
        }

        // Tokenize tin nhắn của người dùng và đưa vào bộ nhớ
        // Format Llama-2 Chat: [INST] {message} [/INST]
        char formatted_prompt[2100];
        sprintf(formatted_prompt, "[INST] %s [/INST]", user_input);
        
        int num_prompt_tokens;
        int* prompt_tokens = (int*)malloc(strlen(formatted_prompt) * sizeof(int) * 4);
        encode(&tokenizer, formatted_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

        printf("ASSISTANT: ");

        // Duyệt qua các token của User (Inference nhưng không in ra)
        for (int i = 0; i < num_prompt_tokens; i++) {
            token = prompt_tokens[i];
            forward(&transformer, token, pos);
            pos++;
        }

        // Bắt đầu Generate câu trả lời của Assistant
        int start_pos = pos;
        while (pos < transformer.config.seq_len) {
            // Dự đoán token tiếp theo
            float* logits = forward(&transformer, token, pos);
            next = sample(&sampler, logits);
            pos++;
            
            if (next == 1 || next == 2) break; // 1: BOS, 2: EOS (Kết thúc câu)

            // In token ra màn hình ngay lập tức (Streaming)
            char* piece = decode(&tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
            token = next;

            // Dừng nếu đạt giới hạn steps cho một câu trả lời
            if (pos - start_pos > steps) break;
        }
        printf("\n");
        free(prompt_tokens);

        if (pos >= transformer.config.seq_len) {
            printf("(Canh bao: Da het bo nho ngu canh, hay go 'reset')\n");
        }
    }

    // --- 4. Giải phóng bộ nhớ ---
    free_run_state(&state);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
```

**Các điểm lưu ý quan trọng:**

1. **Quản lý `pos`**: Đây là chìa khóa. Trong code gốc của Karpathy, `pos`
luôn chạy từ 0 đến `steps`. Ở phiên bản chatbot này, `pos` được giữ nguyên
giá trị sau mỗi lượt chat để mô hình "nhớ" được những gì bạn đã nói ở dòng trên.
2. **Llama-2 Chat Format**: Tôi đã thêm thẻ `[INST]` và `[/INST]`. Nếu bạn
dùng model chưa qua tinh chỉnh (Base model), nó sẽ chỉ viết tiếp văn bản
chứ không trả lời như một trợ lý.
3. **Hàm `safe_printf`**: Đây là hàm có sẵn trong `run.c` dùng để xử lý
các ký tự đặc biệt hoặc byte không hợp lệ trong quá trình decode.
4. **Hiệu năng**: Hãy biên dịch với cờ tối ưu hóa: `gcc -O3 -march=native run.c -lm -o chat`.

> Nếu máy có nhiều nhân CPU, hãy sử dụng OpenMP để tăng tốc: `gcc -O3 -fopenmp run.c -lm -o chat`.

**Cách chạy:**

Sau khi compile, bạn chạy bằng lệnh: `./chat llama2_7b_chat.bin 0.7 0.9 256`

Bạn đã có file checkpoint `.bin` tương thích để nạp vào thử chưa? Nếu chưa,
tôi có thể chỉ bạn cách lấy các file model mini (như 15M hoặc 42M) để test
tốc độ trước.
