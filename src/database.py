import chromadb
from chromadb.config import Settings


class VectorDB:
    def __init__(self, db_path="store/vector_db", collection_name="default_project"):
        """
        初始化向量数据库连接
        :param db_path: 数据库持久化存储路径
        :param collection_name: 集合名称 (用于项目隔离，不同项目的数据互不干扰)
        """
        # print(f" -> [DB] 连接向量数据库: {db_path} | 集合: {collection_name}")

        # 初始化客户端
        self.client = chromadb.PersistentClient(path=db_path)

        # 获取或创建集合 (基于传入的项目名称)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"}  # 使用 L2 欧氏距离
        )

        # 批量写入缓冲区配置
        self.batch_size = 50
        self.buffer_ids = []
        self.buffer_embeddings = []
        self.buffer_metas = []

    def buffer_add(self, unique_id, embedding, metadata):
        """
        将数据添加到内存缓冲区，积攒一定数量后自动写入
        """
        self.buffer_ids.append(unique_id)
        self.buffer_embeddings.append(embedding.tolist())
        self.buffer_metas.append(metadata)

        if len(self.buffer_ids) >= self.batch_size:
            self.flush()

    def flush(self):
        """
        强制将缓冲区的数据写入磁盘 (在处理结束时调用)
        """
        if not self.buffer_ids:
            return

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
        """返回当前集合的数据总量"""
        return self.collection.count()

    def search(self, query_embedding, limit=5, where=None):
        """
        在数据库中搜索最相似的人脸
        :param query_embedding: 目标人脸的特征向量
        :param limit: 返回结果数量
        :param where: 过滤条件字典, e.g. {"data_level": "track"} 或 {"video_name": "xxx.mp4"}
        """
        # 确保输入是 list 格式
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,  # 透传过滤条件
            include=["metadatas", "distances", "documents"]
        )

        # 解析结果为更友好的格式
        parsed_results = []
        # ChromaDB 返回的是二维列表 (支持批量查询)，我们只取第一个 query 的结果
        if results.get('ids') and len(results['ids']) > 0:
            count = len(results['ids'][0])
            for i in range(count):
                item = {
                    "id": results['ids'][0][i],
                    # 兼容性处理，防止某些版本返回 None
                    "distance": results['distances'][0][i] if results.get('distances') else 0.0,
                    "meta": results['metadatas'][0][i] if results.get('metadatas') else {}
                }
                parsed_results.append(item)

        return parsed_results