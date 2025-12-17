import chromadb
from chromadb.config import Settings


class VectorDB:
    def __init__(self, db_path="store/vector_db", collection_name="video_faces"):
        """初始化数据库连接"""
        print(f" -> [DB] 连接向量数据库: {db_path}")
        # 获取持久化客户端
        self.client = chromadb.PersistentClient(path=db_path)
        # 获取或创建集合 (使用 L2 欧氏距离)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"}
        )
        self.batch_size = 50
        self.buffer_ids = []
        self.buffer_embeddings = []
        self.buffer_metas = []

    def buffer_add(self, unique_id, embedding, metadata):
        """添加到缓冲区"""
        self.buffer_ids.append(unique_id)
        # ChromaDB 要求 embedding 是 list 类型
        self.buffer_embeddings.append(embedding.tolist())
        self.buffer_metas.append(metadata)

        # 缓冲区满，自动写入
        if len(self.buffer_ids) >= self.batch_size:
            self.flush()

    def flush(self):
        """强制将缓冲区数据写入数据库"""
        if not self.buffer_ids:
            return

        print(f" -> [DB] 归档批次: {len(self.buffer_ids)} 条数据...")
        self.collection.add(
            ids=self.buffer_ids,
            embeddings=self.buffer_embeddings,
            metadatas=self.buffer_metas
        )
        # 清空缓冲区
        self.buffer_ids = []
        self.buffer_embeddings = []
        self.buffer_metas = []

    def count(self):
        """返回当前库里的数据量"""
        return self.collection.count()

    def search(self, query_embedding, limit=5):
        """
        在数据库中搜索最相似的人脸
        """
        # 确保输入是 list 格式
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        # === 修改点 1: 显式指定 include 参数，防止默认不返回数据 ===
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "distances", "documents"]
        )

        # === 修改点 2: 打印原始返回结果 (关键 Debug) ===
        # print(f"[Debug] Raw Query Results: {results}")
        # (先注释掉上一行，以免刷屏，如果下面修复无效再打开)

        parsed_results = []

        # === 修改点 3: 更健壮的解析逻辑 ===
        # 检查 ids 是否存在且非空
        if results.get('ids') and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
            count = len(results['ids'][0])
            for i in range(count):
                # 以此类推，防止某个字段缺失
                item = {
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i] if results.get('distances') else 0.0,
                    "meta": results['metadatas'][0][i] if results.get('metadatas') else {}
                }
                parsed_results.append(item)
        else:
            # 如果解析失败，打印警告
            print(f"[Warn] 数据库查询返回了空结果。Results keys: {results.keys()}")
            if results.get('ids'):
                print(f"[Warn] IDs structure: {results['ids']}")

        return parsed_results
