using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    /// <summary>
    /// 情绪数据类
    /// </summary>
    public class SentimentData
    {
        /// <summary>
        /// 情绪文本
        /// LoadColumn 特性，文本文件中字段的索引，其描述了每个字段的数据文件顺序
        /// </summary>
        [LoadColumn(0)]
        public string SentimentText;

        /// <summary>
        /// 情绪好坏
        /// LoadColumn 特性，文本文件中字段的索引，其描述了每个字段的数据文件顺序
        /// </summary>
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    /// <summary>
    /// 情绪预测类
    /// </summary>
    public class SentimentPrediction : SentimentData
    {
        /// <summary>
        /// 好坏预测
        /// </summary>
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        /// <summary>
        /// 可能值
        /// </summary>
        public float Probability { get; set; }

        /// <summary>
        /// 分值
        /// </summary>
        public float Score { get; set; }
    }
}
