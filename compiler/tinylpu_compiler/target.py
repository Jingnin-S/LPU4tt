from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Target:
    target_id: str = "tinylpu-experimental-p2-d3-64b"
    lanes: int = 2
    depth: int = 3
    a_ports: int = 1
    b_ports: int = 2
    scratchpad_bytes: int = 64
    buffers: int = 2
    max_k: int = 64
    max_rows: int = 16

    def validate(self):
        if self.lanes not in (1, 2, 4) or self.depth not in (2, 3, 4):
            raise ValueError("Unsupported lane/pipeline target")
        if not 1 <= self.a_ports <= 4 or not 1 <= self.b_ports <= 4:
            raise ValueError("Invalid operand read-port budget")
        if not 1 <= self.scratchpad_bytes <= 65536 or self.buffers not in (1, 2):
            raise ValueError("Invalid data capacity/buffer count")
        if not 1 <= self.max_k <= 64 or not 1 <= self.max_rows <= 16:
            raise ValueError("Unsupported kernel limits")

    def to_dict(self):
        return asdict(self)

    def layout(self, rows, k):
        self.validate()
        if not 1 <= rows <= self.max_rows or not 1 <= k <= self.max_k:
            raise ValueError("Kernel shape outside target limits")
        if min(self.lanes, rows) > self.b_ports:
            raise ValueError("Insufficient weight read ports")
        row_bytes = (k + 1) // 2
        weight_bytes = ((rows + self.lanes - 1) // self.lanes) * self.lanes * row_bytes
        position = weight_bytes
        slots = []
        for _ in range(self.buffers):
            a_addr = position
            position += row_bytes
            position = (position + 1) // 2 * 2
            y_addr = position
            position += rows * 2
            slots.append({"a_addr": a_addr, "y_addr": y_addr})
        if position > self.scratchpad_bytes:
            raise ValueError("Kernel buffers exceed scratchpad capacity")
        return {"w_addr": 0, "weight_reserved_bytes": weight_bytes,
                "slots": slots, "reserved_bytes": position}

    def tile_rows(self, output_rows, k):
        legal = []
        for rows in range(1, min(output_rows, self.max_rows) + 1):
            try:
                self.layout(rows, k)
                legal.append(rows)
            except ValueError:
                pass
        if not legal:
            raise ValueError("No legal output-row tile fits target")
        return max(legal)
