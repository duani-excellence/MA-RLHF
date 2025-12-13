class BlockTable:
    """逻辑块表管理 - 仅负责分页资源管理"""

    def __init__(self, page_size: int, num_pages: int):
        self.page_size = page_size
        self.num_pages = num_pages
        self.free_pages = list(range(num_pages))
        self.allocated_pages = set()

        self.page_usage = [0] * num_pages
        self.next_page = [-1] * num_pages

        # self.

    def _allocate_pages(self, num_pages: int, parent_block_id=-1) -> list[int]: 
        """分配指定数量的页"""
        if len(self.free_pages) < num_pages:
            return []

        allocated = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self.allocated_pages.update(allocated)

        # 初始化块状态
        for page_id in allocated:
            self.page_usage[page_id] = 0
            self.next_page[page_id] = -1

        if parent_block_id != -1:
            self.next_page[parent_block_id] = allocated[0]

        return allocated

    def _free_pages(self, page_ids: list[int]):
        """释放页"""
        for page_id in page_ids:
            if page_id in self.allocated_pages:
                self.allocated_pages.remove(page_id)
                self.free_pages.append(page_id)
                self.page_usage[page_id] = 0
                self.next_page[page_id] = -1

    def get_free_count(self) -> int:
        """获取空闲块数量"""
        return len(self.free_pages)
