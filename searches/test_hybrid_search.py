import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- 1. Dữ liệu văn bản mẫu (Corpus) ---
corpus = [
    "Trí tuệ nhân tạo đang thay đổi thế giới công nghệ.", # AI, technology
    "Học máy là một nhánh quan trọng của trí tuệ nhân tạo.", # Machine Learning, AI
    "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng người.", # NLP, computer understanding
    "Python là ngôn ngữ lập trình phổ biến cho học máy.", # Python, programming, ML
    "Tìm kiếm thông tin là ứng dụng phổ biến của xử lý ngôn ngữ tự nhiên.", # Information retrieval, NLP application
    "Mô hình ngôn ngữ lớn đang rất phát triển gần đây.", # LLM, development
    "Công nghệ AI tạo sinh thu hút nhiều sự chú ý.", # Generative AI, attention
    "Cách mạng công nghiệp 4.0 với cốt lõi là AI và dữ liệu lớn.", # Industry 4.0, AI, Big Data
    "Lập trình Python cho khoa học dữ liệu.", # Python programming, data science
    "Ngành công nghệ thông tin tuyển dụng nhiều kỹ sư AI." # IT industry, AI engineers recruitment
]

# --- 2. Truy vấn tìm kiếm ---
query = "Các ứng dụng của AI trong việc hiểu ngôn ngữ"
# Query này chứa "AI", "ứng dụng", "hiểu ngôn ngữ" - liên quan đến NLP, Information Retrieval, AI applications

print(f"Corpus gồm {len(corpus)} tài liệu.")
print(f"Truy vấn: '{query}'")
print("-" * 30)

# === Phần 3: Sparse Embedding (TF-IDF) ===
print("Đang tính toán Sparse Embeddings (TF-IDF)...")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus) # Sparse matrix
query_vector_sparse = vectorizer.transform([query]) # Sparse matrix

# Tính độ tương đồng Cosine cho Sparse
cosine_similarities_sparse = cosine_similarity(query_vector_sparse, tfidf_matrix).flatten()

# === Phần 4: Dense Embedding (Sentence-BERT) ===
print("Đang tải mô hình Dense Embedding (Sentence-BERT)...")
# Sử dụng mô hình đa ngôn ngữ hỗ trợ tiếng Việt
# paraphrase-multilingual-mpnet-base-v2 là một lựa chọn mạnh
# distiluse-base-multilingual-cased-v2 nhẹ hơn một chút
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)
print(f"Đã tải mô hình: {model_name}")

# Mã hóa corpus và query thành dense vectors
corpus_embeddings_dense = model.encode(corpus, convert_to_tensor=False) # numpy array
query_embedding_dense = model.encode([query], convert_to_tensor=False) # numpy array

# Tính độ tương đồng Cosine cho Dense
# cosine_similarity cũng hoạt động với numpy arrays
cosine_similarities_dense = cosine_similarity(query_embedding_dense, corpus_embeddings_dense).flatten()

# === Phần 5: Kết hợp Kết quả (Hybrid Search - Weighted Average) ===
print("Kết hợp kết quả (Hybrid Search)...")

# Trọng số cho sparse và dense (có thể điều chỉnh)
# alpha = 0.5 nghĩa là đánh giá ngang nhau
alpha = 0.5
beta = 1.0 - alpha

# Điểm số kết hợp = alpha * điểm sparse + beta * điểm dense
# Đảm bảo cả hai loại điểm đều trong khoảng [0, 1] hoặc tương đương.
# Cosine similarity thường trong [-1, 1], nhưng với vector không âm (như TF-IDF) hoặc SBERT chuẩn hóa thì thường là [0, 1].
# Giả sử chúng đều trong khoảng [0, 1] để kết hợp đơn giản.
combined_scores = alpha * cosine_similarities_sparse + beta * cosine_similarities_dense

# Lấy chỉ số sắp xếp theo điểm giảm dần
# Sử dụng numpy.argsort cho mảng numpy
sparse_ranked_indices = np.argsort(cosine_similarities_sparse)[::-1]
dense_ranked_indices = np.argsort(cosine_similarities_dense)[::-1]
combined_ranked_indices = np.argsort(combined_scores)[::-1]

# === Phần 6: Hiển thị Kết quả ===
num_results_to_show = 5

print("\n--- Kết quả chỉ dựa trên Sparse (TF-IDF) ---")
for i in range(num_results_to_show):
    idx = sparse_ranked_indices[i]
    print(f"Top {i+1}: Score={cosine_similarities_sparse[idx]:.4f} - \"{corpus[idx]}\"")

print("\n--- Kết quả chỉ dựa trên Dense (Sentence-BERT) ---")
for i in range(num_results_to_show):
    idx = dense_ranked_indices[i]
    print(f"Top {i+1}: Score={cosine_similarities_dense[idx]:.4f} - \"{corpus[idx]}\"")

print("\n--- Kết quả Kết hợp (Hybrid) ---")
for i in range(num_results_to_show):
    idx = combined_ranked_indices[i]
    print(f"Top {i+1}: Combined Score={combined_scores[idx]:.4f} (Sparse: {cosine_similarities_sparse[idx]:.4f}, Dense: {cosine_similarities_dense[idx]:.4f}) - \"{corpus[idx]}\"")

print("-" * 30)