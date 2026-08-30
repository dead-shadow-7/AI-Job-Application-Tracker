"""Starter skill taxonomy.

Weighted toward AI/ML, backend and data engineering — the roles this tracker is
actually being pointed at. It does not need to be exhaustive: Phase 2 resolves
JD text against ``aliases`` first and flags genuinely unseen terms for review
rather than silently minting new canonical skills.

Aliases must be lowercase; the resolver lowercases its input before matching.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SeedSkill:
    name: str
    slug: str
    category: str
    aliases: list[str] = field(default_factory=list)


def _s(name: str, slug: str, category: str, *aliases: str) -> SeedSkill:
    return SeedSkill(name=name, slug=slug, category=category, aliases=list(aliases))


SEED_SKILLS: list[SeedSkill] = [
    # --- Languages ---------------------------------------------------------
    _s("Python", "python", "language", "python3", "py"),
    _s("JavaScript", "javascript", "language", "js", "ecmascript", "es6"),
    _s("TypeScript", "typescript", "language", "ts"),
    _s("Java", "java", "language", "core java"),
    _s("C++", "cpp", "language", "c++", "cpp"),
    _s("C#", "csharp", "language", "c#", ".net c#"),
    _s("Go", "go", "language", "golang"),
    _s("Rust", "rust", "language"),
    _s("SQL", "sql", "language", "ansi sql"),
    _s("Bash", "bash", "language", "shell", "shell scripting", "sh"),
    _s("R", "r", "language"),
    _s("Scala", "scala", "language"),
    _s("Kotlin", "kotlin", "language"),
    # --- Frontend ----------------------------------------------------------
    _s("React", "react", "frontend", "react.js", "reactjs", "react js"),
    _s("Next.js", "nextjs", "frontend", "next.js", "next js", "nextjs"),
    _s("Vue.js", "vuejs", "frontend", "vue", "vue.js", "vuejs"),
    _s("Angular", "angular", "frontend", "angularjs", "angular.js"),
    _s("Svelte", "svelte", "frontend", "sveltekit"),
    _s("HTML", "html", "frontend", "html5"),
    _s("CSS", "css", "frontend", "css3"),
    _s("Tailwind CSS", "tailwind", "frontend", "tailwind", "tailwindcss"),
    _s("Redux", "redux", "frontend", "redux toolkit"),
    _s("Vite", "vite", "frontend"),
    # --- Backend -----------------------------------------------------------
    _s("FastAPI", "fastapi", "backend", "fast api"),
    _s("Django", "django", "backend", "django rest framework", "drf"),
    _s("Flask", "flask", "backend"),
    _s("Node.js", "nodejs", "backend", "node", "node.js", "nodejs"),
    _s("Express.js", "expressjs", "backend", "express", "express.js"),
    _s("Spring Boot", "spring-boot", "backend", "spring", "springboot"),
    _s("GraphQL", "graphql", "backend"),
    _s("REST APIs", "rest-api", "backend", "rest", "restful apis", "rest api"),
    _s("gRPC", "grpc", "backend"),
    _s("Microservices", "microservices", "backend", "microservice architecture"),
    _s("WebSockets", "websockets", "backend", "websocket", "socket.io"),
    # --- Data stores -------------------------------------------------------
    _s("PostgreSQL", "postgresql", "database", "postgres", "psql"),
    _s("MySQL", "mysql", "database"),
    _s("MongoDB", "mongodb", "database", "mongo"),
    _s("Redis", "redis", "database"),
    _s("Elasticsearch", "elasticsearch", "database", "elastic search", "opensearch"),
    _s("SQLite", "sqlite", "database"),
    _s("Cassandra", "cassandra", "database", "apache cassandra"),
    _s("DynamoDB", "dynamodb", "database", "dynamo db"),
    _s("Neo4j", "neo4j", "database"),
    _s("pgvector", "pgvector", "database", "pg vector"),
    # --- Vector / retrieval ------------------------------------------------
    _s("Pinecone", "pinecone", "vector-db"),
    _s("Weaviate", "weaviate", "vector-db"),
    _s("Qdrant", "qdrant", "vector-db"),
    _s("ChromaDB", "chromadb", "vector-db", "chroma"),
    _s("FAISS", "faiss", "vector-db"),
    _s("RAG", "rag", "ai", "retrieval augmented generation", "retrieval-augmented generation"),
    # --- AI / ML -----------------------------------------------------------
    _s("Machine Learning", "machine-learning", "ai", "ml"),
    _s("Deep Learning", "deep-learning", "ai", "dl", "neural networks"),
    _s("PyTorch", "pytorch", "ai", "torch"),
    _s("TensorFlow", "tensorflow", "ai", "tf"),
    _s("scikit-learn", "scikit-learn", "ai", "sklearn", "scikit learn"),
    _s("Keras", "keras", "ai"),
    _s("Hugging Face", "huggingface", "ai", "hugging face", "transformers", "hf"),
    _s("LangChain", "langchain", "ai", "lang chain"),
    _s("LangGraph", "langgraph", "ai", "lang graph"),
    _s("LlamaIndex", "llamaindex", "ai", "llama index", "gpt index"),
    _s("LLM Fine-tuning", "llm-finetuning", "ai", "fine-tuning", "finetuning", "lora", "peft"),
    _s("Prompt Engineering", "prompt-engineering", "ai", "prompting"),
    _s("NLP", "nlp", "ai", "natural language processing"),
    _s("Computer Vision", "computer-vision", "ai", "cv", "opencv", "image processing"),
    _s("MLOps", "mlops", "ai", "ml ops", "model deployment"),
    _s("OpenAI API", "openai-api", "ai", "openai", "gpt-4", "chatgpt api"),
    _s("Anthropic API", "anthropic-api", "ai", "anthropic", "claude api"),
    _s("Agentic AI", "agentic-ai", "ai", "ai agents", "agent frameworks", "multi-agent"),
    _s("Vector Embeddings", "embeddings", "ai", "embeddings", "sentence transformers"),
    # --- Data engineering --------------------------------------------------
    _s("Pandas", "pandas", "data"),
    _s("NumPy", "numpy", "data"),
    _s("Apache Spark", "spark", "data", "spark", "pyspark"),
    _s("Apache Kafka", "kafka", "data", "kafka"),
    _s("Apache Airflow", "airflow", "data", "airflow"),
    _s("dbt", "dbt", "data", "data build tool"),
    _s("ETL", "etl", "data", "elt", "data pipelines"),
    _s("Snowflake", "snowflake", "data"),
    _s("BigQuery", "bigquery", "data", "google bigquery"),
    _s("Data Warehousing", "data-warehousing", "data", "data warehouse"),
    # --- Cloud / infra -----------------------------------------------------
    _s("AWS", "aws", "cloud", "amazon web services", "ec2", "s3", "lambda"),
    _s("Google Cloud", "gcp", "cloud", "gcp", "google cloud platform"),
    _s("Azure", "azure", "cloud", "microsoft azure"),
    _s("Docker", "docker", "devops", "containers", "containerization"),
    _s("Kubernetes", "kubernetes", "devops", "k8s", "eks", "gke"),
    _s("Terraform", "terraform", "devops", "iac", "infrastructure as code"),
    _s("CI/CD", "ci-cd", "devops", "ci/cd", "continuous integration", "github actions", "jenkins"),
    _s("Linux", "linux", "devops", "unix"),
    _s("Nginx", "nginx", "devops"),
    _s("Serverless", "serverless", "devops", "lambda functions", "cloud functions"),
    _s("Prometheus", "prometheus", "devops", "grafana"),
    # --- Practices ---------------------------------------------------------
    _s("Git", "git", "tooling", "github", "gitlab", "version control"),
    _s("Agile", "agile", "practice", "scrum", "kanban"),
    _s("Unit Testing", "unit-testing", "practice", "pytest", "jest", "testing"),
    _s("System Design", "system-design", "practice", "distributed systems"),
    _s("Data Structures & Algorithms", "dsa", "practice", "dsa", "algorithms", "data structures"),
    _s("Code Review", "code-review", "practice"),
    _s("Technical Writing", "technical-writing", "practice", "documentation"),
]
