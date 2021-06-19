using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;


namespace CardioSpike
{
    static public class ConverterExtension
    {
        static public bool IsEmpty(this string value) { return Converter.IsEmpty(value); }
        static public bool IsEmpty(this decimal value) { return Converter.IsEmpty(value); }
        static public bool IsEmpty(this float value) { return Converter.IsEmpty(value); }
        static public bool IsEmpty(this int value) { return Converter.IsEmpty(value); }
        static public bool IsEmpty(this DateTime value) { return Converter.IsEmpty(value); }
        static public bool IsEmpty(this Guid value) { return Converter.IsEmpty(value); }
        static public bool IsEmpty(this XElement value) { return Converter.IsEmpty(value); }

        static public string IsEmpty(this string value, string def) { return Converter.IsEmpty(value, def); }
        static public decimal IsEmpty(this decimal value, decimal def) { return Converter.IsEmpty(value, def); }
        static public float IsEmpty(this float value, float def) { return Converter.IsEmpty(value, def); }
        static public int IsEmpty(this int value, int def) { return Converter.IsEmpty(value, def); }
        static public DateTime IsEmpty(this DateTime value, DateTime def) { return Converter.IsEmpty(value, def); }
        static public Guid IsEmpty(this Guid value, Guid def) { return Converter.IsEmpty(value, def); }

        static public bool HasValue(this string value) { return Converter.HasValue(value); }
        static public bool HasValue(this decimal value) { return Converter.HasValue(value); }
        static public bool HasValue(this float value) { return Converter.HasValue(value); }
        static public bool HasValue(this int value) { return Converter.HasValue(value); }
        static public bool HasValue(this DateTime value) { return Converter.HasValue(value); }
        static public bool HasValue(this Guid value) { return Converter.HasValue(value); }
        static public bool HasValue(this XElement value) { return Converter.HasValue(value); }

        static public string ToString1251(this string s) { return Converter.ToString1251(s); }
    }

    static public class Converter
    {
        //Thread.WorkThread.CurrentCulture = new CultureInfo("en-us");

        public const int _0000 = 0x0;
        public const int _0001 = 0x1;
        public const int _0010 = 0x2;
        public const int _0011 = 0x3;
        public const int _0100 = 0x4;
        public const int _0101 = 0x5;
        public const int _0110 = 0x6;
        public const int _0111 = 0x7;
        public const int _1000 = 0x8;
        public const int _1001 = 0x9;
        public const int _1010 = 0xA;
        public const int _1011 = 0xB;
        public const int _1100 = 0xC;
        public const int _1101 = 0xD;
        public const int _1110 = 0xE;
        public const int _1111 = 0xF;


        public sealed class EmptyImlicitConverter
        {
            public int Int { get { return Converter.EmptyInt; } } //int.MinValue 
            //static public readonly int NewInt = -2000555000;
            public string String { get { return Converter.EmptyString; } }
            public decimal Decimal { get { return Converter.EmptyDecimal; } }
            public float Float { get { return Converter.EmptyFloat; } }
            public DateTime DateTime { get { return Converter.EmptyDateTime; } }
            public Guid Guid { get { return Converter.EmptyGuid; } }
            public XElement XElement { get { return Converter.EmptyXElement; } }

            internal EmptyImlicitConverter()
            {
            }

            static public implicit operator int(EmptyImlicitConverter c)
            {
                return Converter.EmptyInt;
            }
            static public implicit operator string(EmptyImlicitConverter c)
            {
                return Converter.EmptyString;
            }
            static public implicit operator decimal(EmptyImlicitConverter c)
            {
                return Converter.EmptyDecimal;
            }
            static public implicit operator float(EmptyImlicitConverter c)
            {
                return Converter.EmptyFloat;
            }

            static public implicit operator DateTime(EmptyImlicitConverter c)
            {
                return Converter.EmptyDateTime;
            }
            static public implicit operator Guid(EmptyImlicitConverter c)
            {
                return Converter.EmptyGuid;
            }

            static public implicit operator XElement(EmptyImlicitConverter c)
            {
                return Converter.EmptyXElement;
            }

        }
        static public readonly EmptyImlicitConverter Empty = new EmptyImlicitConverter();
        public const int EmptyInt = int.MinValue; //int.MinValue 
        //static public readonly int NewInt = -2000555000;
        public const string EmptyString = null;
        public const decimal EmptyDecimal = decimal.MinValue; //decimal.MinusOne; 
        public const float EmptyFloat = float.MinValue; // -1
        static public readonly DateTime EmptyDateTime = DateTime.MinValue;
        static public readonly Guid EmptyGuid = Guid.Empty;
        static public readonly XElement EmptyXElement = null;


        static public readonly Type TypeOfString = typeof(string);
        static public readonly Type TypeOfInt = typeof(int);
        static public readonly Type TypeOfFloat = typeof(float);
        static public readonly Type TypeOfDecimal = typeof(decimal);
        static public readonly Type TypeOfGuid = typeof(Guid);
        static public readonly Type TypeOfXElement = typeof(XElement);


        //static public readonly Type TypeOfCode = typeof(Code);

        static public bool IsEmpty(int value)
        {
            return value == EmptyInt;
        }
        /*static public bool IsNew(int value)
        {
            return value == NewInt;
        }*/

        static public bool IsEmpty(string value)
        {
            return value == EmptyString;
        }
        static public bool IsEmpty(decimal value)
        {
            return value == EmptyDecimal;
        }
        static public bool IsEmpty(float value)
        {
            return value == EmptyFloat;
        }
        static public bool IsEmpty(DateTime value)
        {
            return value == EmptyDateTime;
        }
        static public bool IsEmpty(Guid value)
        {
            return value == EmptyGuid;
        }
        static public bool IsEmpty(XElement value)
        {
            return value == EmptyXElement;
        }


        static public string IsEmpty(string value, string def)
        {
            return value != EmptyString ? value : def;
        }
        static public decimal IsEmpty(decimal value, decimal def)
        {
            return value != EmptyDecimal ? value : def;
        }
        static public float IsEmpty(float value, float def)
        {
            return value != EmptyFloat ? value : def;
        }
        static public int IsEmpty(int value, int def)
        {
            return value != EmptyInt ? value : def;
        }

        static public DateTime IsEmpty(DateTime value, DateTime def)
        {
            return value != EmptyDateTime ? value : def;
        }
        static public Guid IsEmpty(Guid value, Guid def)
        {
            return value != EmptyGuid ? value : def;
        }

        static public bool HasValue(string value)
        {
            return value != EmptyString;
        }
        static public bool HasValue(decimal value)
        {
            return value != EmptyDecimal;
        }
        static public bool HasValue(float value)
        {
            return value != EmptyFloat;
        }
        static public bool HasValue(int value)
        {
            //return (value != EmptyInt) && (value != NewInt); //так не правильно потому что ключ не может быть пустым
            return value != EmptyInt;
        }
        static public bool HasValue(DateTime value)
        {
            return value != EmptyDateTime;
        }
        static public bool HasValue(Guid value)
        {
            return value != EmptyGuid;
        }
        static public bool HasValue(XElement value)
        {
            return value != EmptyXElement;
        }



        static public string ToString(decimal value)
        {
            if (IsEmpty(value)) return EmptyString;
            return value.ToString().Replace(',', '.');
        }

        static public string ToString(float value)
        {
            if (IsEmpty(value)) return EmptyString;
            return value.ToString("G").Replace(',', '.');
        }

        static public string ToString(int value)
        {
            if (IsEmpty(value)) return EmptyString;
            return value.ToString(); //.Replace(',', '.');
        }

        static public string ToString(Guid value)
        {
            if (IsEmpty(value)) return EmptyString;
            return value.ToString("D").ToUpper(); //.Replace(',', '.');
            //???? возможно и скобочкинадо убирать
        }

        static public string ToString(XElement value)
        {
            if (IsEmpty(value)) return EmptyString;
            return value.ToString();
        }
        static public string ToString(XElement value, SaveOptions options)
        {
            if (IsEmpty(value)) return EmptyString;
            return value.ToString(options);
        }


        static public string ToString(DateTime value)
        {
            if (IsEmpty(value)) return EmptyString;
            //return value.ToString(); //.Replace(',', '.');
            if (value.TimeOfDay == TimeSpan.FromTicks(0))   //в склсервере если убрать составляющую времени то при конвертации стринг-дате-датетайм может получитса не тот формат. надо явно задавать дэйтформат
                //return value.ToString("d"); 
                return value.ToString("s").Substring(0, 10);
            else
                return value.ToString("s");
            //ms-help://MS.MSDNQTR.v90.en/dv_fxfund/html/bb79761a-ca08-44ee-b142-b06b3e2fc22b.htm

        }


        static public string ToStringAsDocName(string value)
        {
            if (IsEmpty(value)) return EmptyString;
            while (value.IndexOf("\n") != -1) value = value.Replace("\n", " ");
            while (value.IndexOf("\r") != -1) value = value.Replace("\r", " ");
            while (value.IndexOf("  ") != -1) value = value.Replace("  ", " ");
            value = value.Trim();
            if (value.Length > 125) value = value.Substring(0, 125) + "...";
            return value;
        }

        static private Encoding _encoding1251; //инициализируем по необходимости
        static private Encoding Encoding1251 { get { return _encoding1251 ?? (_encoding1251 = Encoding.GetEncoding(1251)); } }
        static public string ToString1251(string value)
        {
            if (IsEmpty(value)) return EmptyString;
            Encoding enc = Encoding1251;
            string newvalue = enc.GetString(enc.GetBytes(value.Normalize())).Normalize();
            if (newvalue != value) return newvalue;
            return value;
        }
        /*static public string ToString1251(MemoryStream ms)
        {
            if (ms == null) return EmptyString;
                 
            Encoding enc = Encoding1251;
            enc.

            string newvalue = enc.GetString(enc.GetBytes(value));
            if (newvalue != value) return newvalue;
            return value;
        }*/

        //!!! имя документа должно подвергнутся некой преработке - например




        static public int ToIntRoundDef(string value, int def)
        {
            if (string.IsNullOrEmpty(value)) return def;
            int result;
            if (int.TryParse(value, out result)) return result;
            if (value.IndexOfAny(PointСommaArray) > -1)
            {
                decimal r = ToDecimal(value);
                if (r == EmptyDecimal) return def;
                else return decimal.ToInt32(r);
            }
            return def;
        }
        static public int ToIntRound(string value)
        {
            return ToIntRoundDef(value, EmptyInt);
        }
        static public int ToIntDef(string value, int def)
        {
            if (string.IsNullOrEmpty(value)) return def;
            int result;
            if (int.TryParse(value, out result)) return result;
            return def;
        }
        static public int ToIntDef(object value, int def)
        {
            if (value == null) return def;
            return ToIntDef(value.ToString(), def);
        }
        static public int ToInt(string value)
        {
            return ToIntDef(value, EmptyInt);
        }


        static public readonly char PointCharCurrent = ((decimal)5.5).ToString()[1];
        static public readonly char[] PointСommaArray = new char[] { '.', ',' };

        // поставить преобразователь точки в запятую
        static public decimal ToDecimal(string value)
        {
            if (string.IsNullOrEmpty(value)) return EmptyDecimal;

            if (PointCharCurrent == '.') value = value.Replace(',', '.');
            else
            if (PointCharCurrent == ',') value = value.Replace('.', ',');

            //value = value.Replace('.', PointCharCurrent).Replace(',', PointCharCurrent);

            if (value.IndexOf('E') == -1)
            {
                decimal result;
                if (decimal.TryParse(value, out result)) return result;
                return EmptyDecimal;
                //return decimal.Parse(value.Replace('.', PointCharCurrent));
            }
            else
            {
                double result;
                if (double.TryParse(value, out result)) return (decimal)result;
                return EmptyDecimal;
            }

        }
        static public decimal ToDecimalDef(string value, decimal def)
        {
            decimal result = ToDecimal(value);
            if (Converter.IsEmpty(result)) return def;
            return result;
        }
        static public decimal ToDecimalRound(string value, int round)
        {
            return ToDecimalRoundDef(value, round, EmptyDecimal);
        }
        static public decimal ToDecimalRoundDef(string value, int round, decimal def)
        {
            decimal result = ToDecimal(value);
            if (Converter.IsEmpty(result)) return def;
            return decimal.Round(result, round);
        }



        static public float ToFloat(string value)
        {
            if (string.IsNullOrEmpty(value)) return EmptyFloat;
            return float.Parse(value.Replace('.', PointCharCurrent));
        }

        static public DateTime ToDateTime(string value)
        {
            if (string.IsNullOrEmpty(value)) return EmptyDateTime;
            DateTime result = Converter.EmptyDateTime;
            if (DateTime.TryParse(value, out result)) return result;
            return Converter.EmptyDateTime;
        }

        //https://excel2.ru/articles/kak-excel-hranit-datu-i-vremya
        //дате 25 01 1900 число 25
        //дате 14.01.2011 число 40557

        //test("29.02.1900", 60); есть глючок - эксель не знает что нет такого числа
        //test("01.03.1900", 60); должно быть так, но так не понимаетет
        //test("01.03.1900", 61); но эксель считает так, может только старый, фик знает - короче надо отсчитывать отсюда

        //воттак идет в эксле
        //28.02.1900 -> 59		
        //29.02.1900 -> 60 //этой в реальности даты нет		
        //01.03.1900 -> 61
        //02.03.1900 -> 62


        static private readonly DateTime startExcelDate = new DateTime(1900, 3, 1).Subtract(TimeSpan.FromDays(1)); //потомушта отсчет с единицы
        static public DateTime ToExcelDateTime(double value)
        {
            if (value < 0) return Empty;

            int shift60 = 60;
            if (value <= 60) shift60 -= 1; //исправляем экселевский глючок
            if (value <= 60) throw new NotSupportedException(); //можно заремить и будет работать, но всеравно ограничение надо, пусть такое

            value -= shift60;

            var d = startExcelDate.Add(TimeSpan.FromDays(value));
            return d;
            //return Converter.EmptyDateTime;
        }
        static public double ToExcelDateTime(DateTime value)
        {
            if (value.IsEmpty()) return Empty;

            TimeSpan ts = value.Subtract(startExcelDate);
            double result = ts.TotalDays;

            int shift60 = 60;
            if (result <= 0) shift60 -= 1; //исправляем экселевский глючок
            if (result <= 0) throw new NotSupportedException(); //можно заремить и будет работать, но всеравно ограничение надо, пусть такое

            result += shift60;
            return result;
            //return Converter.EmptyDateTime;
        }
        static public int ToExcelDays(DateTime value)
        {
            if (value.IsEmpty()) return Empty;
            return (int)Math.Truncate(ToExcelDateTime(value));
            //return Converter.EmptyDateTime;
        }

        static public (DateTime, DateTime) MinMax((DateTime, DateTime) date) => date.Item1 <= date.Item2 ? date : (date.Item2, date.Item1);
        static public (DateTime, DateTime) MinMax(DateTime date1, DateTime date2) => MinMax((date1, date2));



        static public Guid ToGuid(string value)
        {
            if (string.IsNullOrEmpty(value)) return EmptyGuid;
            Guid result = Converter.EmptyGuid;
            try
            {
                result = new Guid(value);
                result = new Guid(value);

            }
            catch
            {
                throw new NotImplementedException();
            }
            return result;
        }
        static public XElement ToXElement(string value)
        {
            if (string.IsNullOrEmpty(value)) return EmptyXElement;
            XElement result = Converter.EmptyXElement;
            try
            {
                result = XElement.Parse(value);
            }
            catch
            {
                throw new NotImplementedException();
            }
            return result;
        }



        static public string Crop(string s, int maxlen, string completion)
        {
            if (s == null) return null;
            if (s.Length <= maxlen) return s;
            if (maxlen <= completion.Length) throw new ArgumentOutOfRangeException();

            maxlen -= completion.Length;
            s = s.Substring(0, maxlen);
            if (completion.Length > 0) s = s + completion;
            return s;
        }
        static public string Crop(string s, int maxlen)
        {
            return Crop(s, maxlen, "...");
        }

        static public System.Security.Cryptography.MD5 MD5 = new System.Security.Cryptography.MD5CryptoServiceProvider();
        /*static public string ToMD5Base64String(this string b)
        {
            byte[] bytearray = new byte[b.Length]; //преобразованиев байты возможно нужно сделать както более правильно, может с учетом кодировки
            for (int a = 0; a < b.Length; a++)
                bytearray[a] = (byte)(b[a]);

            byte[] hasharray = MD5.ComputeHash(bytearray); //Encoding.Default.GetBytes(input)
            string result = Convert.ToBase64String(hasharray, Base64FormattingOptions.None);
            return result;
        }*/
        static public string To8000MD5(string s)
        {
            if (s == null) return null;
            if (s.Length > 8000) s = s.Substring(0, 8000);
            byte[] bytearray = Encoding1251.GetBytes(s); //не забываем учесть что md5 в базе вычисляетса от 1251 кодировки
            byte[] hasharray = MD5.ComputeHash(bytearray);
            s = Convert.ToBase64String(hasharray, Base64FormattingOptions.None);
            return s;
        }

        /*
        Thread.WorkThread.CurrentUICulture.NumberFormat.NumberDecimalSeparator = ".";
        Thread.WorkThread.CurrentUICulture.NumberFormat.CurrencyDecimalSeparator = ".";
        */

        static public string tojson(this object obj)
        {
            string jsontext = JsonConvert.SerializeObject(obj, Formatting.Indented);
            return jsontext;
        }

    }


    static public class TypeOf<T>
    {
        public static readonly Type Value = typeof(T);
    }



}

