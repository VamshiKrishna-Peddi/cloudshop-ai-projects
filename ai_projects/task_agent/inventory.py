"""Validated inventory arithmetic shared by the CLI, tools, and browser UI."""
import csv
import io
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InventoryItem:
    product: str
    stock: int
    reorder_level: int

    @property
    def order_quantity(self):
        return max(0, self.reorder_level - self.stock)


def parse_inventory(text: str):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')), strict=True)
    required = {'product', 'stock', 'reorder_level'}
    try:
        headers = reader.fieldnames
        if not headers or not required.issubset(headers) or len(headers) != len(set(headers)):
            raise ValueError('CSV needs unique product, stock, and reorder_level columns.')
        items, seen = [], set()
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f'Row {line}: incorrect number of columns.')
            product = row['product'].strip()
            if not product or '\n' in product or '\r' in product or product.casefold() in seen:
                raise ValueError(f'Row {line}: product must be nonempty, unique, and on one line.')
            values = [row[key].strip() for key in ('stock', 'reorder_level')]
            if any(not re.fullmatch(r'[0-9]{1,12}', value) for value in values):
                raise ValueError(f'Row {line}: stock and reorder_level must be nonnegative whole numbers (up to 12 digits).')
            items.append(InventoryItem(product, *map(int, values)))
            seen.add(product.casefold())
        if not items:
            raise ValueError('CSV must contain at least one inventory row.')
        return items
    except csv.Error as exc:
        raise ValueError('Invalid CSV formatting.') from exc


def render_report(items):
    low = [item for item in items if item.order_quantity > 0]
    return '# Inventory reorder report\n\n' + '\n'.join(
        f'- {item.product}: stock {item.stock}; order {item.order_quantity}' for item in low
    ) + f'\n\nProducts requiring reorder: {len(low)}\n'
