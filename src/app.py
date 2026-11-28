# -*- coding: utf-8 -*-
"""FastAPI 服务：银行电话营销响应预测。

启动（在项目根目录执行）：
    python -m uvicorn src.app:app --reload

然后浏览器打开：
    http://127.0.0.1:8000/docs      ← 在这里可以直接点按钮测试接口
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_one

app = FastAPI(
    title="银行电话营销响应预测 API",
    description="输入客户信息，返回其认购定期存款的概率。用于营销名单排序，降低外呼成本。",
    version="1.0.0",
)

# 业务阈值：概率高于此值的客户，建议优先拨打（可按外呼预算调整）
PRIORITY_THRESHOLD = 0.30


class Customer(BaseModel):
    """客户信息。字段默认值是一个常见客户，方便在 /docs 里直接测试。"""

    age: int = Field(35, ge=18, le=100, description="年龄")
    job: str = Field("admin.", description="职业")
    marital: str = Field("married", description="婚姻状况")
    education: str = Field("university.degree", description="学历")
    default: str = Field("no", description="是否有违约记录")
    housing: str = Field("yes", description="是否有房贷")
    loan: str = Field("no", description="是否有个人贷款")
    contact: str = Field("cellular", description="联系方式")
    month: str = Field("may", description="最后联系月份")
    day_of_week: str = Field("mon", description="最后联系星期")
    campaign: int = Field(2, ge=1, description="本次活动中联系次数")
    pdays: int = Field(999, ge=0, description="距上次联系天数（999=未联系过）")
    previous: int = Field(0, ge=0, description="之前联系次数")
    poutcome: str = Field("nonexistent", description="之前营销结果")

    emp_var_rate: float = Field(1.1, alias="emp.var.rate", description="就业变化率")
    cons_price_idx: float = Field(93.994, alias="cons.price.idx", description="消费者价格指数")
    cons_conf_idx: float = Field(-36.4, alias="cons.conf.idx", description="消费者信心指数")
    euribor3m: float = Field(4.857, description="欧元区 3 个月利率")
    nr_employed: float = Field(5191.0, alias="nr.employed", description="就业人数（千人）")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="认购概率（0~1）")
    level: str = Field(..., description="概率等级：高 / 中 / 低")
    suggestion: str = Field(..., description="营销建议")


def _to_raw_dict(customer: Customer) -> dict:
    """把请求对象还原成训练时使用的原始字段名。"""
    return {
        "age": customer.age,
        "job": customer.job,
        "marital": customer.marital,
        "education": customer.education,
        "default": customer.default,
        "housing": customer.housing,
        "loan": customer.loan,
        "contact": customer.contact,
        "month": customer.month,
        "day_of_week": customer.day_of_week,
        "campaign": customer.campaign,
        "pdays": customer.pdays,
        "previous": customer.previous,
        "poutcome": customer.poutcome,
        "emp.var.rate": customer.emp_var_rate,
        "cons.price.idx": customer.cons_price_idx,
        "cons.conf.idx": customer.cons_conf_idx,
        "euribor3m": customer.euribor3m,
        "nr.employed": customer.nr_employed,
    }


@app.get("/", summary="健康检查")
def root():
    return {"service": "bank-marketing-response", "status": "ok"}


@app.post("/predict", response_model=PredictionResponse, summary="预测单个客户的认购概率")
def predict(customer: Customer):
    probability = predict_one(_to_raw_dict(customer))

    if probability >= PRIORITY_THRESHOLD:
        level, suggestion = "高", "建议优先拨打"
    elif probability >= PRIORITY_THRESHOLD / 3:
        level, suggestion = "中", "可放入第二轮名单"
    else:
        level, suggestion = "低", "建议暂缓拨打"

    return PredictionResponse(
        probability=round(probability, 4),
        level=level,
        suggestion=suggestion,
    )
