# gen_dataset_v2.py
import random
import os

# ============ 变量池 ============
KIN = ["妈", "爸", "奶奶", "爷爷", "姥姥", "姥爷", "叔叔", "阿姨"]
AMOUNT = ["一万", "两万", "三万", "五万", "八万", "十万"]
URGENCY = ["赶紧", "马上", "立刻", "抓紧", "尽快"]
SECRET = ["千万别告诉", "不要告诉", "别跟你", "先别声张", "保密"]
PLATFORMS = ["淘宝", "京东", "顺丰", "邮政", "移动"]

# ============ 诈骗：铺垫段 ============
OPENINGS_FRAUD = {
    "冒充亲属": [
        "喂，{kin}，是我啊，你现在方便说话吗？",
        "喂，{kin}，我这边有点事想跟你说一下。",
        "{kin}，你最近身体还好吧？我这边有点急事。",
    ],
    "冒充公检法": [
        "您好，请问是{kin}吗？我是北京市公安局的民警，工号0731。",
        "您好，我这边是法院执行局的，有一个案子需要跟您核实一下。",
    ],
    "冒充客服": [
        "您好，请问是{kin}吗？我是{platform}的客服，工号1234。",
        "您好，打扰一下，我是{platform}售后服务中心的。",
    ],
    "中奖/补贴": [
        "您好，恭喜您成为我们平台的幸运用户！",
        "您好，我们这边有一笔补贴需要发放给您。",
    ],
    "诱导远程操控": [
        "您好，我是{platform}的安全专员，检测到您的账户有异常。",
        "您好，您的账户存在风控风险，需要协助您处理一下。",
    ],
    "养老/投资": [
        "阿姨您好，我是社区养老服务中心的，有个好消息要告诉您。",
        "叔叔您好，我们这边有个养老投资项目，专门给咱们老年人准备的。",
    ],
    "刷单": [
        "您好，我看到您之前浏览过我们的兼职信息，现在还有兴趣吗？",
        "您好，我们这边有个在家就能做的兼职，您想了解一下吗？",
    ],
    "冒充老师/领导": [
        "喂，{kin}，我是孩子班主任王老师，您现在方便吗？",
        "{kin}，我是你领导，现在方便说话吗？",
    ],
}

# ============ 诈骗：核心段 ============
CORE_FRAUD = {
    "冒充亲属": [
        "我{time}出事了，现在在医院，急需{amt}手术费，你{urgency}转钱过来。",
        "我被人控制了，对方要{amt}，你{urgency}转账，{SECRET}别人。",
        "我撞了人，现在需要{amt}私了，{SECRET}我爸妈。",
    ],
    "冒充公检法": [
        "您的身份证涉嫌一起洗钱案件，现在需要您配合调查。",
        "您的银行账户涉案被冻结，需要将{amt}转入安全账户进行核查。",
    ],
    "冒充客服": [
        "您之前有一笔订单出现异常，我们给您办理退款，需要您提供银行卡和验证码。",
        "您的快递丢失，可以进行双倍理赔，需要您配合操作一下。",
    ],
    "中奖/补贴": [
        "您中奖了，奖金是{amt}，但需要先缴纳一部分税费才能领取。",
        "您可以领取一笔{amt}的补贴，请点击链接填写银行卡信息。",
    ],
    "诱导远程操控": [
        "您的账户存在异常交易，请下载我们的APP，开屏幕共享，我帮您核查。",
        "需要您配合开启屏幕共享，我帮您把钱转到安全账户。",
    ],
    "养老/投资": [
        "我们有个养老基地投资项目，保本高收益，{urgency}投入{amt}。",
        "您之前的理财出现亏损，需要缴纳保证金帮您追回。",
    ],
    "刷单": [
        "刷单返利，投入越多赚得越多，{urgency}转账{amt}激活任务。",
        "先转账{amt}完成任务，本金和佣金一起返还给您。",
    ],
    "冒充老师/领导": [
        "孩子在学校摔伤了，现在急需{amt}手术费，{urgency}转账。",
        "我这边有个急事，你先帮我转笔钱，{SECRET}别人。",
    ],
}

# ============ 诈骗：收尾段 ============
CLOSINGS_FRAUD = [
    "{SECRET}任何人，这事只有你能帮我。",
    "你{urgency}操作，晚了就来不及了。",
    "记住，{SECRET}家人，我处理好再告诉他们。",
    "这事不能声张，你按我说的做就行。",
    "现在就把钱转过来，账户我稍后发给你。",
]

# ============ 正常：三段 ============
OPENINGS_NORMAL = [
    "喂，{kin}，是我。",
    "您好，请问是{kin}吗？",
    "{kin}，你现在忙不忙？",
]

CORE_NORMAL = [
    "我今晚加班，晚饭不回去吃了。",
    "您的快递到了，我放门口了。",
    "明天社区有免费体检，您记得去。",
    "您的银行卡积分可以兑换礼品。",
    "我们这边新出了一款理财产品，收益稳定。",
    "您的宽带月底到期，可以续费了。",
    "孩子最近学习怎么样？我周末带他出去玩。",
    "您上次说的那个事，我帮您问了一下。",
]

CLOSINGS_NORMAL = [
    "那就这样，我先挂了啊。",
    "您注意身体，有事再给我打电话。",
    "好的，祝您生活愉快。",
    "没问题的话就这样，再见。",
]

# ============ 工具函数 ============
def fill(t):
    """填充模板中的变量占位符"""
    t = t.replace("{kin}", random.choice(KIN))
    t = t.replace("{amt}", random.choice(AMOUNT))
    t = t.replace("{urgency}", random.choice(URGENCY))
    t = t.replace("{SECRET}", random.choice(SECRET))
    t = t.replace("{time}", random.choice(["刚刚", "刚才", "现在", "今天"]))
    t = t.replace("{platform}", random.choice(PLATFORMS))
    return t


# ============ 拼接生成对话 ============
def generate_fraud(fraud_type):
    """生成一段完整的诈骗电话文本"""
    opening = random.choice(OPENINGS_FRAUD[fraud_type])
    core = random.choice(CORE_FRAUD[fraud_type])
    closing = random.choice(CLOSINGS_FRAUD)

    parts = [fill(opening), fill(core), fill(closing)]

    # 50% 概率插入一句闲聊
    if random.random() < 0.5:
        small_talk = random.choice([
            "你最近身体还好吧？",
            "这两天天气冷了，注意保暖啊。",
            "我这两天一直想给你打电话来着。",
        ])
        parts.insert(1, small_talk)

    return " ".join(parts)


def generate_normal():
    """生成一段正常的电话/对话文本"""
    parts = [
        fill(random.choice(OPENINGS_NORMAL)),
        fill(random.choice(CORE_NORMAL)),
        fill(random.choice(CLOSINGS_NORMAL)),
    ]
    return " ".join(parts)


# ============ 主函数 ============
def generate_dataset(output_path="dataset/test_dataset.txt",
                     n_per_fraud_type=12, n_normal=96):
    samples = []

    # 诈骗样本
    for fraud_type in OPENINGS_FRAUD.keys():
        for _ in range(n_per_fraud_type):
            text = generate_fraud(fraud_type)
            samples.append((text, 1, fraud_type))

    # 正常样本
    for _ in range(n_normal):
        text = generate_normal()
        samples.append((text, 0, "正常"))

    random.shuffle(samples)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 主文件（两列）
    with open(output_path, "w", encoding="utf-8") as f:
        for text, label, _ in samples:
            f.write(f"{text}\t{label}\n")

    # 分层文件（三列，带类别）
    with open(output_path.replace(".txt", "_with_type.txt"), "w", encoding="utf-8") as f:
        for text, label, ftype in samples:
            f.write(f"{text}\t{label}\t{ftype}\n")

    print(f"✅ 生成完成：{len(samples)} 条")
    print(f"   诈骗：{sum(1 for _, l, _ in samples if l == 1)} 条")
    print(f"   正常：{sum(1 for _, l, _ in samples if l == 0)} 条")
    print(f"   主文件：{output_path}")
    print(f"   分层文件：{output_path.replace('.txt', '_with_type.txt')}")


if __name__ == "__main__":
    generate_dataset()