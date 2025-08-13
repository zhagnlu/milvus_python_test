import random
import string
import json
import numpy as np
from datetime import datetime, timedelta
import uuid

class EnhancedJSONGenerator:
    """增强的JSON数据生成器"""
    
    def __init__(self):
        self.counter = 0
        
    def random_string(self, length=6, charset=None):
        """生成随机字符串"""
        if charset is None:
            charset = string.ascii_lowercase + string.digits
        return ''.join(random.choices(charset, k=length))
    
    def random_email(self):
        """生成随机邮箱"""
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com']
        username = self.random_string(random.randint(5, 15))
        domain = random.choice(domains)
        return f"{username}@{domain}"
    
    def random_phone(self):
        """生成随机电话号码"""
        return f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    def random_date(self, start_date=None, end_date=None):
        """生成随机日期"""
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 12, 31)
        
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        return random_date.isoformat()
    
    def random_address(self):
        """生成随机地址"""
        streets = ['Main St', 'Oak Ave', 'Pine Rd', 'Elm St', 'Maple Dr', 'Cedar Ln']
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
        states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA']
        
        street_num = random.randint(100, 9999)
        street = random.choice(streets)
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = random.randint(10000, 99999)
        
        return {
            "street": f"{street_num} {street}",
            "city": city,
            "state": state,
            "zip_code": str(zip_code),
            "country": "USA"
        }
    
    def random_user_profile(self, user_id):
        """生成随机用户档案"""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
        
        profile = {
            "user_id": user_id,
            "personal_info": {
                "first_name": random.choice(first_names),
                "last_name": random.choice(last_names),
                "email": self.random_email(),
                "phone": self.random_phone(),
                "date_of_birth": self.random_date(datetime(1960, 1, 1), datetime(2000, 12, 31)),
                "gender": random.choice(["male", "female", "other"]),
                "address": self.random_address()
            },
            "preferences": {
                "language": random.choice(["en", "es", "fr", "de", "zh", "ja"]),
                "timezone": random.choice(["UTC-8", "UTC-5", "UTC+0", "UTC+1", "UTC+8"]),
                "theme": random.choice(["light", "dark", "auto"]),
                "notifications": {
                    "email": random.choice([True, False]),
                    "sms": random.choice([True, False]),
                    "push": random.choice([True, False])
                }
            },
            "account": {
                "created_at": self.random_date(datetime(2018, 1, 1), datetime(2023, 12, 31)),
                "last_login": self.random_date(datetime(2023, 1, 1), datetime(2024, 12, 31)),
                "status": random.choice(["active", "inactive", "suspended"]),
                "verification_level": random.randint(1, 5),
                "subscription_tier": random.choice(["free", "basic", "premium", "enterprise"])
            }
        }
        return profile
    
    def random_product_data(self, product_id):
        """生成随机产品数据"""
        categories = ['electronics', 'clothing', 'books', 'home', 'sports', 'beauty', 'automotive']
        brands = ['Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'LG', 'Canon', 'Dell']
        
        product = {
            "product_id": product_id,
            "basic_info": {
                "name": f"Product {product_id}",
                "sku": f"SKU-{product_id:06d}",
                "category": random.choice(categories),
                "brand": random.choice(brands),
                "model": f"Model-{self.random_string(4)}",
                "description": f"This is a description for product {product_id}"
            },
            "pricing": {
                "base_price": round(random.uniform(10.0, 1000.0), 2),
                "sale_price": round(random.uniform(5.0, 800.0), 2),
                "currency": "USD",
                "discount_percentage": random.randint(0, 50),
                "tax_rate": round(random.uniform(0.0, 0.15), 3)
            },
            "inventory": {
                "stock_quantity": random.randint(0, 1000),
                "reserved_quantity": random.randint(0, 100),
                "reorder_point": random.randint(10, 50),
                "supplier_id": random.randint(1, 20),
                "warehouse_location": random.choice(["A1", "B2", "C3", "D4"])
            },
            "ratings": {
                "average_rating": round(random.uniform(1.0, 5.0), 1),
                "total_reviews": random.randint(0, 1000),
                "rating_distribution": {
                    "5_star": random.randint(0, 500),
                    "4_star": random.randint(0, 300),
                    "3_star": random.randint(0, 150),
                    "2_star": random.randint(0, 100),
                    "1_star": random.randint(0, 50)
                }
            },
            "specifications": {
                "dimensions": {
                    "length": round(random.uniform(1.0, 100.0), 2),
                    "width": round(random.uniform(1.0, 100.0), 2),
                    "height": round(random.uniform(1.0, 100.0), 2),
                    "weight": round(random.uniform(0.1, 50.0), 2)
                },
                "features": random.sample([
                    "waterproof", "wireless", "bluetooth", "wifi", "touchscreen",
                    "rechargeable", "portable", "durable", "lightweight", "eco-friendly"
                ], random.randint(2, 6)),
                "warranty_months": random.randint(0, 60)
            }
        }
        return product
    
    def random_order_data(self, order_id):
        """生成随机订单数据"""
        order = {
            "order_id": order_id,
            "customer_info": {
                "customer_id": random.randint(1000, 9999),
                "name": f"Customer {order_id}",
                "email": self.random_email(),
                "phone": self.random_phone(),
                "shipping_address": self.random_address(),
                "billing_address": self.random_address()
            },
            "order_details": {
                "order_date": self.random_date(datetime(2023, 1, 1), datetime(2024, 12, 31)),
                "status": random.choice(["pending", "confirmed", "shipped", "delivered", "cancelled"]),
                "payment_method": random.choice(["credit_card", "paypal", "bank_transfer", "cash"]),
                "payment_status": random.choice(["pending", "paid", "failed", "refunded"]),
                "shipping_method": random.choice(["standard", "express", "overnight"]),
                "tracking_number": f"TRK{random.randint(100000000, 999999999)}"
            },
            "items": [
                {
                    "product_id": random.randint(1, 100),
                    "quantity": random.randint(1, 10),
                    "unit_price": round(random.uniform(10.0, 500.0), 2),
                    "total_price": round(random.uniform(10.0, 5000.0), 2),
                    "discount": round(random.uniform(0.0, 50.0), 2)
                }
                for _ in range(random.randint(1, 5))
            ],
            "totals": {
                "subtotal": round(random.uniform(50.0, 2000.0), 2),
                "tax": round(random.uniform(5.0, 200.0), 2),
                "shipping": round(random.uniform(5.0, 50.0), 2),
                "discount": round(random.uniform(0.0, 100.0), 2),
                "total": round(random.uniform(50.0, 2500.0), 2)
            }
        }
        return order
    
    def random_log_data(self, log_id):
        """生成随机日志数据"""
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        services = ['web-server', 'database', 'auth-service', 'payment-service', 'notification-service']
        ip_addresses = ['192.168.1.1', '10.0.0.1', '172.16.0.1', '127.0.0.1']
        
        log = {
            "log_id": log_id,
            "timestamp": self.random_date(datetime(2024, 1, 1), datetime(2024, 12, 31)),
            "level": random.choice(log_levels),
            "service": random.choice(services),
            "message": f"Log message {log_id} with some details",
            "context": {
                "request_id": str(uuid.uuid4()),
                "user_id": random.randint(1, 10000),
                "session_id": str(uuid.uuid4()),
                "ip_address": random.choice(ip_addresses),
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "endpoint": random.choice(['/api/users', '/api/products', '/api/orders', '/api/auth'])
            },
            "performance": {
                "response_time_ms": random.randint(10, 5000),
                "memory_usage_mb": round(random.uniform(10.0, 1000.0), 2),
                "cpu_usage_percent": round(random.uniform(1.0, 100.0), 2),
                "database_queries": random.randint(1, 50)
            },
            "error_details": {
                "error_code": random.randint(100, 599),
                "error_type": random.choice(['ValidationError', 'DatabaseError', 'NetworkError', 'TimeoutError']),
                "stack_trace": f"Error stack trace for log {log_id}",
                "recovered": random.choice([True, False])
            } if random.choice(log_levels) in ['ERROR', 'CRITICAL'] else None
        }
        return log
    
    def random_analytics_data(self, event_id):
        """生成随机分析数据"""
        event_types = ['page_view', 'button_click', 'form_submit', 'purchase', 'download']
        page_names = ['home', 'products', 'cart', 'checkout', 'profile', 'search']
        
        analytics = {
            "event_id": event_id,
            "event_type": random.choice(event_types),
            "timestamp": self.random_date(datetime(2024, 1, 1), datetime(2024, 12, 31)),
            "user": {
                "user_id": random.randint(1, 10000),
                "session_id": str(uuid.uuid4()),
                "device_type": random.choice(['desktop', 'mobile', 'tablet']),
                "browser": random.choice(['chrome', 'firefox', 'safari', 'edge']),
                "os": random.choice(['windows', 'macos', 'linux', 'ios', 'android']),
                "country": random.choice(['US', 'CN', 'JP', 'DE', 'UK', 'FR']),
                "city": random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
            },
            "page": {
                "url": f"https://example.com/{random.choice(page_names)}",
                "title": f"Page Title {event_id}",
                "referrer": random.choice(['google.com', 'facebook.com', 'twitter.com', 'direct']),
                "load_time_ms": random.randint(100, 5000)
            },
            "interaction": {
                "element_id": f"element_{random.randint(1, 100)}",
                "element_type": random.choice(['button', 'link', 'form', 'image']),
                "coordinates": {
                    "x": random.randint(0, 1920),
                    "y": random.randint(0, 1080)
                },
                "scroll_depth": random.randint(0, 100)
            },
            "conversion": {
                "goal_completed": random.choice([True, False]),
                "goal_name": random.choice(['signup', 'purchase', 'download', 'contact']),
                "revenue": round(random.uniform(0.0, 1000.0), 2) if random.choice([True, False]) else None,
                "funnel_step": random.randint(1, 5)
            }
        }
        return analytics
    
    def generate_complex_json(self, data_type="mixed", complexity="medium"):
        """生成复杂JSON数据"""
        if data_type == "user_profile":
            return self.random_user_profile(self.counter)
        elif data_type == "product":
            return self.random_product_data(self.counter)
        elif data_type == "order":
            return self.random_order_data(self.counter)
        elif data_type == "log":
            return self.random_log_data(self.counter)
        elif data_type == "analytics":
            return self.random_analytics_data(self.counter)
        else:
            # 混合类型
            types = ["user_profile", "product", "order", "log", "analytics"]
            return getattr(self, f"random_{random.choice(types)}_data")(self.counter)
    
    def generate_batch(self, count, data_type="mixed", complexity="medium"):
        """生成一批JSON数据"""
        batch = []
        for i in range(count):
            self.counter = i
            batch.append(self.generate_complex_json(data_type, complexity))
        return batch

# 使用示例
if __name__ == "__main__":
    generator = EnhancedJSONGenerator()
    
    # 生成不同类型的测试数据
    print("=== 用户档案数据 ===")
    user_data = generator.generate_complex_json("user_profile")
    print(json.dumps(user_data, indent=2, ensure_ascii=False))
    
    print("\n=== 产品数据 ===")
    product_data = generator.generate_complex_json("product")
    print(json.dumps(product_data, indent=2, ensure_ascii=False))
    
    print("\n=== 订单数据 ===")
    order_data = generator.generate_complex_json("order")
    print(json.dumps(order_data, indent=2, ensure_ascii=False))
    
    print("\n=== 日志数据 ===")
    log_data = generator.generate_complex_json("log")
    print(json.dumps(log_data, indent=2, ensure_ascii=False))
    
    print("\n=== 分析数据 ===")
    analytics_data = generator.generate_complex_json("analytics")
    print(json.dumps(analytics_data, indent=2, ensure_ascii=False)) 