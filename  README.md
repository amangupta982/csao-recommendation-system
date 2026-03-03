# AI-Powered Cart Add-On Recommendation System (CSAO)

An enterprise-style contextual recommendation system designed to increase Average Order Value (AOV) in food delivery platforms through intelligent cart add-on suggestions.

Built for Zomathon Hackathon 2026.

---

## 🚀 Problem

Food delivery users typically order only main dishes, ignoring complementary items like beverages, desserts, or sides.

This leads to:
- Low Average Order Value (AOV)
- Missed cross-sell opportunities
- Generic, non-personalized suggestions

We solve this using a real-time AI-powered contextual ranking system.

---

## 🏗 System Architecture

The system follows a **Two-Stage Retrieval + Ranking Framework**:

### 1️⃣ Candidate Generation
- Same restaurant filtering
- Popularity-based filtering
- Candidate shortlisting (Top 20–50)

### 2️⃣ Contextual Ranking (XGBoost)
Uses:
- User behavior features
- Cart sequence features
- Time-based context
- Price affinity features
- Interaction features

### 3️⃣ Bandit Re-Ranking Layer
- Thompson Sampling
- Exploration vs Exploitation
- Online adaptive learning

---

## 🧠 Key Features

✔ Sequential Cart Modeling  
✔ Context-Aware Ranking  
✔ Multi-Armed Bandit Optimization  
✔ Session-Based Evaluation  
✔ Latency Simulation (<50ms)  
✔ Scalability Design (Kubernetes + Stateless Pods)

---

## 📊 Model Performance

Sample Output:

- AUC: ~0.84  
- Precision@1: ~0.58  
- Precision@5: 0.20 (bounded by design)  
- NDCG: ~0.90  
- Simulated Inference Latency: ~30–40ms  

Precision@5 is bounded at 0.20 since each session contains one relevant item.

We prioritize:
- Precision@1 (Top suggestion quality)
- NDCG (Ranking quality)

---

## 📂 Dataset

This project uses:

1️⃣ Zomato Bangalore Restaurants Dataset  
https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants

2️⃣ Synthetic Session Simulation  
Since real cart-session data is proprietary, we simulate realistic cart behavior using:
- Real restaurant ratings
- Real price distributions
- Context-aware time sampling
- Sequential cart generation

This allows reproducible evaluation of ranking metrics.

Note: Dataset files are not included in this repository due to size limits.

---

## ⚡ How To Run

Clone the repository:
git clone https://github.com/amangupta982/csao-recommendation-system.git

cd csao-recommendation-system
Install dependencies:
pip install -r requirements.txt

Run:
---

## 🏎 Latency Design

Component-wise simulated latency:

- Feature Fetch: ~8 ms  
- Candidate Retrieval: ~15 ms  
- Contextual Ranking: ~10 ms  
- Bandit Re-Ranking: ~5 ms  

Total Inference Latency: < 50 ms

Designed for real-time serving.

---

## 📈 Scalability Strategy

Assumptions:
- 10M sessions/day
- Peak ~1000 QPS

Architecture:
- Stateless inference pods
- Kubernetes autoscaling
- Load balancing
- Redis feature store
- Offline storage (S3 / BigQuery concept)

Estimated 10–20 replicas handle peak safely.

---

## 💰 Business Impact Estimation

Assumptions:
- 10M daily sessions
- ₹120 average add-on value
- 5% attach rate improvement

Incremental revenue:
≈ ₹6 Crore/day  
≈ ₹2190 Crore/year

Even small improvements scale massively.

---

## 🔮 Future Enhancements

- Transformer-based cart encoder
- Real-time A/B testing framework
- Reinforcement Learning optimization
- Segment-aware personalization

---

## 👥 Team

Elite Crew  
Aman Gupta – Data Science  
Rohan Kumar Mandal – AIML  
Shahista Aleem – Data Science  

---

## 🏁 Conclusion

This project demonstrates a production-style, scalable, real-time contextual recommendation system with:

- Strong offline metrics
- Online learning capability
- Latency-aware design
- Business impact modeling

Built to simulate real-world deployment readiness.