# -*- coding: utf-8 -*-
"""课后报告生成器"""

from app.ai.behavior_analyzer import BEHAVIOR_LABELS


def generate_report_data(snapshots: list) -> dict:
    """
    根据快照数据生成课后报告的统计信息。

    Args:
        snapshots: AttentionSnapshot列表

    Returns:
        报告数据字典
    """
    if not snapshots:
        return {"error": "没有数据"}

    scores = [s.attention_score for s in snapshots]
    times = [s.elapsed_seconds for s in snapshots]

    # 基本统计
    avg_score = round(sum(scores) / len(scores), 1)
    max_score = max(scores)
    min_score = min(scores)

    # 找到专注度最低谷时段（连续3个最低分的起点）
    low_periods = []
    window = 3
    if len(scores) >= window:
        for i in range(len(scores) - window + 1):
            avg_window = sum(scores[i:i + window]) / window
            low_periods.append((times[i], avg_window))
        low_periods.sort(key=lambda x: x[1])

    # 行为总体分布
    total_behaviors = {}
    for s in snapshots:
        if s.behavior_counts:
            for beh, count in s.behavior_counts.items():
                total_behaviors[beh] = total_behaviors.get(beh, 0) + count

    behavior_display = {
        BEHAVIOR_LABELS.get(k, k): v for k, v in total_behaviors.items()
    }

    # 生成改进建议
    suggestions = []
    if avg_score < 60:
        suggestions.append("整体专注度偏低，建议增加师生互动环节（提问、小组讨论）")
    if low_periods:
        worst_time = low_periods[0][0]
        minutes = worst_time // 60
        suggestions.append(f"第{minutes}分钟左右专注度最低，建议在此处插入案例演示或休息")
    if total_behaviors.get("head_down", 0) > total_behaviors.get("focused", 0) * 0.3:
        suggestions.append("低头比例较高，可能存在玩手机现象，建议增加课堂互动")

    if not suggestions:
        suggestions.append("课堂表现良好，继续保持当前教学节奏")

    return {
        "avg_score": avg_score,
        "max_score": max_score,
        "min_score": min_score,
        "total_snapshots": len(snapshots),
        "duration_minutes": round((times[-1] - times[0]) / 60, 1) if len(times) > 1 else 0,
        "behavior_distribution": behavior_display,
        "low_periods": low_periods[:3] if low_periods else [],
        "suggestions": suggestions,
        "timeline": {
            "times": times,
            "scores": scores,
        },
    }
