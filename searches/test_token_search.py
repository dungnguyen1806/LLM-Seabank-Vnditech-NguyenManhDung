import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Dữ liệu văn bản mẫu (Corpus) ---
corpus = [
    "Trí tuệ nhân tạo đang thay đổi thế giới công nghệ.",
    "Học máy là một nhánh quan trọng của trí tuệ nhân tạo.",
    "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng người.",
    "Python là ngôn ngữ lập trình phổ biến cho học máy.",
    "Tìm kiếm thông tin là ứng dụng phổ biến của xử lý ngôn ngữ tự nhiên.",
    "Mô hình ngôn ngữ lớn đang rất phát triển gần đây.",
    "Công nghệ AI tạo sinh thu hút nhiều sự chú ý."
]

# --- 2. Khởi tạo TfidfVectorizer ---
# TfidfVectorizer sẽ xây dựng từ điển và tính toán trọng số TF-IDF
# Kết quả trả về là một ma trận thưa (sparse matrix)
vectorizer = TfidfVectorizer()

# --- 3. Tạo Sparse Embeddings (TF-IDF) cho Corpus ---
# Fit: Xây dựng từ điển (vocabulary) và tính IDF từ corpus
# Transform: Chuyển đổi từng văn bản trong corpus thành vector TF-IDF (sparse embedding)
tfidf_matrix = vectorizer.fit_transform(corpus)

# In ra kích thước của ma trận TF-IDF (số tài liệu x số từ trong từ điển)
# và xem một phần của vector đầu tiên (dưới dạng sparse)
print(f"Kích thước ma trận TF-IDF (sparse): {tfidf_matrix.shape}")
print("Vector TF-IDF (sparse) của tài liệu đầu tiên:\n", tfidf_matrix[0])
print("\nTừ điển (features):", vectorizer.get_feature_names_out()) # In ra các từ trong từ điển

# --- 4. Truy vấn tìm kiếm ---
query = "ứng dụng trí tuệ nhân tạo trong xử lý ngôn ngữ"

# --- 5. Tạo Sparse Embedding (TF-IDF) cho Truy vấn ---
# Sử dụng vectorizer ĐÃ ĐƯỢC FIT trên corpus để transform truy vấn
# Điều này đảm bảo truy vấn được biểu diễn trong cùng không gian vector với corpus
query_vector = vectorizer.transform([query])

print("\nVector TF-IDF (sparse) của truy vấn:\n", query_vector)

# --- 6. Tính toán độ tương đồng (Cosine Similarity) ---
# Tính độ tương đồng cosine giữa vector truy vấn và tất cả các vector tài liệu trong corpus
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# --- 7. Xếp hạng và Hiển thị kết quả ---
# Lấy chỉ số của các tài liệu được sắp xếp theo độ tương đồng giảm dần
# argsort trả về chỉ số từ thấp đến cao, nên dùng [::-1] để đảo ngược
related_docs_indices = cosine_similarities.argsort()[::-1]

print(f"\n--- Kết quả tìm kiếm cho truy vấn: '{query}' ---")
# In ra top 3 tài liệu tương đồng nhất
num_results = 3
for i in range(num_results):
    doc_index = related_docs_indices[i]
    similarity_score = cosine_similarities[doc_index]
    # Chỉ hiển thị nếu độ tương đồng > 0
    if similarity_score > 0:
        print(f"Top {i+1}: Score={similarity_score:.4f} - \"{corpus[doc_index]}\"")