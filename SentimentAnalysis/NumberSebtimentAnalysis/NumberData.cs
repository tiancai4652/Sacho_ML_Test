using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumberSebtimentAnalysis
{
    public class NumberData
    {
        [LoadColumn(0)]
        public string Key { get; set; }

        [LoadColumn(1),ColumnName("Lable")]
        public bool Value { get; set; }
    }

    public class NumberDataPrediction : NumberData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
