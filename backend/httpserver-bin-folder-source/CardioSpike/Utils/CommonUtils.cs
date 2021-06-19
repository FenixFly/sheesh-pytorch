using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO;
using System.Data;
using System.Data.SqlClient;
using System.Net;
//using System.Web;
using System.Xml;
using System.Xml.Linq;
using System.Collections;
using System.Collections.Specialized;
using System.Runtime.InteropServices;

namespace CardioSpike
{

    //[SuppressUnmanagedCodeSecurity]
    /*internal static class Win32Native
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool Beep(int frequency, int duration);
    }*/

    /*static public class CommonExtensions
    {
        static public bool Contains(this string s, string substring)
        {
            if (string.IsNullOrEmpty(s)) return false;
            if (string.IsNullOrEmpty(substring)) return false;
            return s.IndexOf(substring) != -1;
        }
    }*/
    public enum ReplaceCount { Once = 1, OnceLast = -1, All = 0, Total = int.MaxValue };

    static public class IsNullExtension
    {
        static public string NullIfEmpty(this string s)
        {
            return (s.IsNullOrEmpty() ? null : s);
        }

        static public string IsNull(this string s)
        {
            return (s ?? "");
        }

        static public string IsNull(this string s, string def)
        {
            return (s ?? def);
        }

        static public bool IsNullOrEmpty(this string s)
        {
            return (s ?? "") == "";
        }

        static public bool IsNullOrWhiteSpace(this string value)
        {
            if (value != null)
            {
                for (int i = 0; i < value.Length; i++)
                {
                    if (!char.IsWhiteSpace(value[i]))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        //static public bool IfNullOrEmpty(this string s, string def)
        //{
        //    return string.IsNullOrEmpty(s) ? def : s;
        //}

        //static public bool HasValue(this XAttribute a)
        //{
        //    return a.IsNullOrEmpty();
        //}
        //static public bool HasValue(this XElement e)
        //{
        //    return e.IsNullOrEmpty();
        //}

        static public bool IsNullOrEmpty(this XAttribute a)
        {
            return a == null ? true : a.Value.IsNullOrEmpty();
        }
        static public bool IsNullOrEmpty(this XElement e)
        {
            return e == null ? true : e.Value.IsNullOrEmpty();
        }

        static public string IsNull(this XAttribute a)
        {
            return a.IsNullOrEmpty() ? "" : a.Value.IsNull();
        }
        static public string IsNull(this XElement e)
        {
            return e.IsNullOrEmpty() ? "" : e.Value.IsNull();
        }

        static public string IsNull(this XAttribute a, string def)
        {
            return a.IsNullOrEmpty() ? def : a.Value;
        }
        static public string IsNull(this XElement e, string def)
        {
            return e.IsNullOrEmpty() ? def : e.Value;
        }

    }


    static public class CommonUtils
    {
        static public void test()
        {
            return;
        }

        static public string PutDownParams(string s, string paramlist)
        {
            // втакой функции нельзя чтобы параметр стоял в самом переди без пробела и чтобы они шли друг за другом
            if (!s.Contains("##ParamByName[")) return s;
            //if (s.Contains("##ParamByName[##ParamByName[")) throw new Exception("некорректно заданы параметры в PutDownParams");
            //if (s.Contains("]##]##")) throw new Exception("некорректно заданы параметры в PutDownParams");

            string[] list = s.Split(new string[] { "##ParamByName[", "]##" }, StringSplitOptions.None); // если 2 параметра идут друг за другом то должна оставатса пустая строка

            //for (int i = 1; i < cmdlist.Length - 2; i++)
            //{
            //    if (cmdlist[i - 1] != "##ParamByName[") continue; 
            //    if (cmdlist[i + 1] != "]##") continue; 
            //    string priceElement = cmdlist[i];
            //    if ((priceElement.Length > 50) || (priceElement.Contains(" ")) || (priceElement.Contains("\node")) || (priceElement.Contains("\priceElement"))) continue;
            //    priceElement = CommonUtils.GetParamByName(priceElement, nameValueParamlist).Trim();
            //    cmdlist[i] = priceElement;
            //    cmdlist[i - 1] = "";
            //    cmdlist[i + 1] = "";
            //}

            for (int i = 1; i < list.Length; i += 2)
            {
                string t = list[i].Trim();

                //в случае такой остановки, в результате всеравно не останется текстовых блоков { "##ParamByName[", "]##" }
                if (t == "") break; //throw new Exception("некорректно заданы параметры в PutDownParams");

                if ((t.Length > 50) || (t.Contains(" ")) || (t.Contains("\n")) || (t.Contains("\t"))) continue;
                t = CommonUtils.GetParamByName(t, paramlist).Trim();
                list[i] = t;
            }

            return string.Concat(list);
        }


        static public string StrToBase64Str(string str)
        {
            if (str == null) return null;
            str = str.Trim();
            if (str == "") return "";

            string source_b64 = Convert.ToBase64String(System.Text.ASCIIEncoding.Default.GetBytes(str.ToCharArray()));
            return source_b64;
        }
        static public string StrFromBase64Str(string str_b64)
        {
            if (str_b64 == null) return null;
            str_b64 = str_b64.Trim();
            if (str_b64 == "") return "";

            byte[] result_b = Convert.FromBase64CharArray(str_b64.ToCharArray(), 0, str_b64.Length);
            string result = System.Text.ASCIIEncoding.Default.GetString(result_b);
            return result;
        }

        static private char findnumcharM64(char c)
        {
            char[,] mmm = new char[,] { { '1', 'a' }, { '2', 'b' }, { '3', 'c' }, { '4', 'd' }, { '5', 'e' }, { '6', 'f' }, { '7', 'g' }, { '8', 'h' }, { '9', 'i' } };
            c = char.ToLower(c);
            for (int a = 0; a < mmm.Length; a++)
                if (mmm[a, 0] == c) return mmm[a, 1];
                else if (mmm[a, 1] == c) return mmm[a, 0];

            throw new ArgumentOutOfRangeException("с");
        }
        static public string EncodeStrM64(string s, int len)
        {
            if ((len < 0) || (len > 9)) throw new ArgumentOutOfRangeException("len");

            try
            {
                for (int a = 1; a <= len; a++)
                    s = StrToBase64Str(s);

                if (len == 0) //автоматический режим
                    do
                    {
                        s = StrToBase64Str(s);
                        len++;
                    }
                    while (s.EndsWith("="));

                char c = findnumcharM64(len.ToString()[0]);
                int pos = s.Length / 2;
                if (char.IsUpper(s[pos])) c = char.ToUpper(c); //маскируем регистр
                s = s.Insert(pos, c.ToString());

                return s;
            }
            catch (Exception e)
            {
                throw new Exception("неудача при запаковке строки", e);
            }
        }

        static public string DecodeStrM64(string s)
        {
            if (s.IsNullOrEmpty() || s.Length == 1) throw new ArgumentOutOfRangeException("s");

            try
            {
                int pos = s.Length / 2;
                char l = s[pos];
                s = s.Remove(pos, 1);

                l = findnumcharM64(l);
                int len = int.Parse(l.ToString());

                for (int a = 1; a <= len; a++)
                    s = StrFromBase64Str(s);

                return s;
            }
            catch //внутреннее исключение не показываем
            {
                throw new Exception("неудача при распаковке строки");
            }
        }


        // блок xml команд
        public static string GetNodeParam(XmlNode Node, string Name)
        {
            if (Node == null) return string.Empty;
            if (Node.Attributes == null) return string.Empty;
            //Name = Name.ToUpper(); //может на чемто сказатса

            XmlNode iddAttribute = Node.Attributes.GetNamedItem(Name);
            if (iddAttribute == null)
                return string.Empty;
            else
                return iddAttribute.Value;
        }
        public static string GetNodeParam(XElement Node, string Name)
        {
            return Node.Attribute(Name).IsNull();
        }

        public static void SetNodeParam(XmlNode Node, string name, string value)
        {
            if (Node == null) return;
            if (Node.Attributes == null) return;
            name = name.ToUpper();

            XmlNode iddAttribute = Node.Attributes.GetNamedItem(name);

            if (iddAttribute == null)
                iddAttribute = Node.OwnerDocument.CreateAttribute(name);
            else
                Node.Attributes.RemoveNamedItem(name);

            iddAttribute.Value = value;
            Node.Attributes.SetNamedItem(iddAttribute);
        }

        public static void SetNodeParam(XElement Node, string name, string value)
        {
            if (Node == null) return;
            value = value.IsNull();

            name = name.ToUpper();
            XAttribute attr = Node.Attribute(name);
            if (attr == null)
            {
                attr = new XAttribute(name, value);
                Node.Add(attr);
            }
            else
            {
                attr.Value = value;
            }

            if (value.IsNullOrEmpty())
                attr.Remove();
        }

        public static string GetChildNodeText(XmlNode node, string childName)
        {
            if (node == null) return null;
            for (int a = 0; a < node.ChildNodes.Count; a++)
                if (node.ChildNodes[a].Name == childName)
                    return (node.ChildNodes[a].InnerText) ?? string.Empty;
            return string.Empty;
        }
        public static string GetChildNodeParam(XmlNode node, string childName, string childParamName)
        {
            if (node == null) return null;
            for (int a = 0; a < node.ChildNodes.Count; a++)
                if (node.ChildNodes[a].Name == childName)
                    return GetNodeParam(node.ChildNodes[a], childParamName);
            return string.Empty;
        }

        public static string GetNodeText(XmlNode Node)
        {
            if (Node == null) return string.Empty;
            return Node.InnerText ?? string.Empty;
        }
        public static void SetNodeText(XmlNode Node, string value)
        {
            if (Node == null) return;
            Node.InnerText = value != null ? value : string.Empty;
        }
        // конец блока xml команд




        //static private readonly char[] AttributeSpecSymbols = new char[] { '&', '<', '>', '"' };
        static internal string MakeAttributeValue(string paramValue)
        {
            //if (paramValue.IndexOfAny(AttributeSpecSymbols == -1) return paramValue;
            paramValue = paramValue.Replace("&", "&amp;");
            paramValue = paramValue.Replace("<", "&lt;");
            paramValue = paramValue.Replace(">", "&gt;");
            paramValue = paramValue.Replace("\"", "&quot;");
            return paramValue;
        }
        static private string MakeAttribute(string paramName, string paramValue) //пока нигде не юзаетса 
        {
            if (string.IsNullOrEmpty(paramName)) return string.Empty;
            if (paramValue == null) return string.Empty;
            paramValue = paramValue.Trim();
            if (paramValue == string.Empty) return string.Empty;

            //if (paramValue.IndexOfAny(AttributeSpecSymbols) != -1) 
            paramValue = MakeAttributeValue(paramValue);
            return string.Concat(" ", paramName, "=\"", paramValue, "\"");
        }
        static public void MakeAttribute(StringBuilder builder, string paramName, string paramValue) //юзаетса в b2 app_code
        {
            if (string.IsNullOrEmpty(paramName)) return;
            if (paramValue == null) return;
            paramValue = paramValue.Trim();
            if (paramValue == string.Empty) return;

            //if (paramValue.IndexOfAny(AttributeSpecSymbols) != -1) 
            paramValue = MakeAttributeValue(paramValue);

            builder.Append(" ");
            builder.Append(paramName);
            builder.Append("=\"");
            builder.Append(paramValue);
            builder.Append("\"");
        }

        public static void Beep(int frequency, int duration)
        {
            System.Console.Beep(frequency, duration);
            /*
            if ((frequency < 0x25) || (frequency > 0x7fff)) throw new ArgumentOutOfRangeException("frequency", frequency, "Beep from 37 to 32767 hertz");
            if (duration == 0) return;
            if (duration < 0) throw new ArgumentOutOfRangeException("duration", duration, "Beep greater 0 milliseconds");
            Win32Native.Beep(frequency, duration);
            */
        }

        static public string XmlToIndentString(XmlDocument xml)
        {
            // приводим в красивый вид
            XmlWriter writer = null;
            StringBuilder sb = new StringBuilder();
            try
            {
                XmlWriterSettings settings = new XmlWriterSettings();
                settings.Indent = true;
                settings.IndentChars = ("\t");
                settings.OmitXmlDeclaration = true;
                writer = XmlWriter.Create(sb, settings);
                writer.WriteNode(xml.CreateNavigator(), true);
                writer.Flush();
            }
            finally
            {
                if (writer != null)
                    writer.Close();
            }

            return sb.ToString();
        }


        /*static public string isNull(string s)
        {
            if (s == null)
                return "";
            else
                return s;
        }

        static public string isNull(string s, string defaultValue)
        {
            if (s == null)
                return defaultValue;
            else
                return s;
        }*/

        //следует использовать IsNulOrEmpty


        public static string isNull(object o)
        {
            return o == null ? string.Empty : o.ToString();
        }

        public static string isNull(string s)
        {
            return s == null ? string.Empty : s;
        }

        public static string isNull(string s, string v)
        {
            return s == null ? v : s;
        }

        public static object isNull(ref object o, System.Type t)
        {
            return o != null ? o : o = Activator.CreateInstance(t);
        }

        public static T isNull<T>(T o, T def) where T : class
        {
            return o != null ? o : def;
        }


        public static T isNullTo<T>(object o, T isNotNull) where T : struct
        {
            if (o == null) return isNotNull;

            try
            {
                if (typeof(T).IsEnum)
                    return (T)Enum.Parse(typeof(T), o.ToString(), true);

                return (T)Convert.ChangeType(o, typeof(T));
            }
            catch (Exception e)
            {
                throw new Exception(string.Format("{2} Не удалось конвертировать [{0}] в [{1}]", o, typeof(T), e.Message));
            }
        }

        static public int StrToIntDef(string s, int def)
        {
            //if (s == null) return def;
            int result;
            if (int.TryParse(s, out result))
                return result;
            else
                return def;
        }

        static public Guid StrToGuidDef(string s, Guid def)
        {
            if (string.IsNullOrEmpty(s)) return def;

            // потом это можно ускорить отдельно сделав все в трех циклах
            s = s.Trim();
            int l = s.Length;
            if ((l != 32) && (l != 36) && (l != 38)) return def;
            /*
                      10        20        30        
            01234567890123456789012345678901234567890
            dddddddddddddddddddddddddddddddd
            dddddddd-dddd-dddd-dddd-dddddddddddd
            {dddddddd-dddd-dddd-dddd-dddddddddddd} 
            */
            char c;
            for (int a = 0; a < l; a++)
            {
                c = s[a];
                if (l == 38)
                {
                    if (a == 0) if (c == '{') continue; else return def;
                    if ((a == 9) || (a == 14) || (a == 19) || (a == 24)) if (c == '-') continue; else return def;
                    if (a == 37) if (c == '}') continue; else return def;
                }
                else if (l == 36)
                {
                    if ((a == 8) || (a == 13) || (a == 18) || (a == 23)) if (c == '-') continue; else return def;
                }

                if (((c >= '0') && (c <= '9')) || ((c >= 'a') && (c <= 'f')) || ((c >= 'A') && (c <= 'Z'))) continue;

                return def;
            }

            //return new Guid(s);
            try
            {
                return new Guid(s);
            }
            catch
            {
                return def;
            }
        }

        static public double StrToDoubleDef(string s, double def)
        {
            double result;

            if (double.TryParse(s, out result))
                return result;
            else
                return def;
        }

        static public string GetString(DataRow row, string col)
        {
            if (row == null) return string.Empty;
            int i = row.Table.Columns.IndexOf(col);
            if (i == -1) return string.Empty;

            object o = row[i];
            return CommonUtils.isNull(o as string);
        }
        static public int GetInt(DataRow row, string col, int def)
        {
            if (row == null) return def;
            int i = row.Table.Columns.IndexOf(col);
            if (i == -1) return def;

            object o = row[i];
            if (o is int) return (int)o;
            return CommonUtils.StrToIntDef(o as string, def);
        }

        /*static public int StrToIntDefxxx(string s, int def)
        {					            
            s = s.Trim();
            if (s.Length == 0) return def;

            for(int a = 0; a < s.Length; a++)
                if (!Char.IsNumber(s, a))
                    return def;
            try 
            { 
                return Int32.ServerParse(s); 
            }
            catch
            { 
                throw new Exception("ошибка в StrToIntDef"); 
                //return def;
            }
        }*/

        static public string CleanIndent(string s)
        {
            if (s == null) return null;
            //StringBuilder result = new StringBuilder(s); 
            if (s.Contains('\r')) s = s.Replace('\r', ' ');
            if (s.Contains('\n')) s = s.Replace('\n', ' ');
            if (s.Contains('\t')) s = s.Replace('\t', ' ');
            s = s.Trim();
            while (s.IndexOf("  ") > -1) s = s.Replace("  ", " ");
            return s.Trim();
        }

        static public string ParamlistToUrl(string paramlist)
        {
            paramlist = ImproveParamlist(paramlist);
            return "##url##?"
                    + (CommonUtils.ImproveParamlist(paramlist) + "|")
                        .Remove(0, 2)
                        .Replace("||", "&")
                        .Replace("&|", "");
        }

        static public string Str2Utf8Url(string s)
        {
            byte[] data = Encoding.UTF8.GetBytes(s);
            StringBuilder sb = new StringBuilder(data.Length * 3);
            for (int a = 0; a < data.Length; a++)
            {
                sb.Append('%' + data[a].ToString("X"));
            }
            return sb.ToString();
        }

        static public string DateTimeSQLToDiman(string d)
        {
            if (d.Length > 0)
                return DateTimeSQLToDiman(DateTime.Parse(d));
            else
                return "";
        }
        static public string DateTimeSQLToDiman(DateTime d)
        {
            return d.ToString(@"dd\/MM\/yyyy\ HH\:mm");
        }

        static public string ImproveParamlist(string paramlist)
        {
            StringBuilder sb = new StringBuilder("||" + paramlist + "||");
            while (sb.ToString() != sb.ToString().Replace("|||", "||")) sb.Replace("|||", "||");
            return sb.ToString();
        }

        static public string MakeParamParam(string paramName, int paramValue)
        {
            if (string.IsNullOrEmpty(paramName)) return "";
            if (paramValue <= 0) return "";

            return "||" + paramName.ToUpper() + "=" + paramValue.ToString();
        }
        static public string MakeParamParam(string paramName, string paramValue)
        {
            if (string.IsNullOrEmpty(paramName)) return "";
            if (string.IsNullOrEmpty(paramValue)) return "";

            return "||" + paramName.ToUpper() + "=" + paramValue;
        }

        static public string MakeParamParam(string paramName, DataRow row)
        {
            return MakeParamParam(paramName, GetParamByName(paramName, row));
        }

        static public string MakeParamError(string paramName, string error)
        {
            if (error == "") return "";
            return "||ERROR:" + paramName.ToUpper() + "=" + error + ". ";
        }

        static public string SetParamParam(string paramName, string paramValue, string paramlist)
        {
            if (paramName == "") return paramlist;
            if (paramValue == "") return RemoveParamParam(paramName, paramlist);

            if (!paramlist.StartsWith("||")) paramlist = "||" + paramlist;
            if (!paramlist.EndsWith("||")) paramlist = paramlist + "||";

            //if ((nameValueParamlist.Substring(0,2) != "||")||((nameValueParamlist.Substring(nameValueParamlist.Length-2,2) != "||")))
            //    nameValueParamlist = "||" + nameValueParamlist + "||";

            paramName = paramName.ToUpper();

            if (paramlist.IndexOf("||" + paramName + "=") > -1)
                return CommonUtils.MakeParamParam(paramName, paramValue) + CommonUtils.RemoveParamParam(paramName, paramlist);
            else
                return CommonUtils.MakeParamParam(paramName, paramValue) + paramlist;
        }

        static public string RemoveParamParam(string paramName, string paramlist)
        {
            if (!paramlist.StartsWith("||")) paramlist = "||" + paramlist;
            if (!paramlist.EndsWith("||")) paramlist = paramlist + "||";

            //if ((nameValueParamlist.Substring(0,2) != "||")||((nameValueParamlist.Substring(nameValueParamlist.Length-2,2) != "||")))
            //    nameValueParamlist = "||" + nameValueParamlist + "||";

            string spath = "||" + paramName.ToUpper() + "=";
            string spath1 = "||" + paramName.ToLower() + "=";
            string spath2 = "||" + paramName + "=";

            int b = paramlist.IndexOf(spath);
            if (b == -1) b = paramlist.IndexOf(spath1);
            if (b == -1) b = paramlist.IndexOf(spath2);

            if (b == -1) return paramlist;

            int e = paramlist.IndexOf("||", b + 3);

            while ((b > -1) && (e > -1))
            {
                paramlist = paramlist.Remove(b, e - b);

                b = paramlist.IndexOf(spath);
                if (b == -1) b = paramlist.IndexOf(spath1);
                e = paramlist.IndexOf("||", b + 3);
            }

            return paramlist;
        }

        static public string GetParamByName(string paramName, string paramlist)
        {
            if (!paramlist.StartsWith("||")) paramlist = "||" + paramlist;
            if (!paramlist.EndsWith("||")) paramlist = paramlist + "||";
            //nameValueParamlist = "||" + nameValueParamlist + "||";

            string spath = "||" + paramName.ToUpper() + "=";
            string spath1 = "||" + paramName.ToLower() + "=";
            string spath2 = "||" + paramName + "=";

            int b = paramlist.IndexOf(spath);
            if (b == -1) b = paramlist.IndexOf(spath1);
            if (b == -1) b = paramlist.IndexOf(spath2);

            if (b == -1) return "";

            //SaveLogs("b + l=" + (b + spath.Length).ToString());

            paramlist = paramlist.Remove(0, b + spath.Length);

            int e = paramlist.IndexOf("||");
            if (e < 0) return "";

            return paramlist.Substring(0, e);
        }

        static public string GetParamByName(string paramName, DataRow row)
        {
            if (paramName == null) return "";
            if (row == null) return "";
            if (paramName == "") return "";

            if (row.Table.Columns.Contains(paramName))
                if (row[paramName] != null)
                    if (row[paramName].ToString() != "")
                        return row[paramName].ToString();
            return "";
        }

        static public int GetIntParamByName(string paramName, string paramlist)
        {
            string s = GetParamByName(paramName, paramlist);
            return StrToIntDef(s, -1);
        }
        static public int GetIntParamByName(string paramName, string paramlist, int def)
        {
            string s = GetParamByName(paramName, paramlist);
            return StrToIntDef(s, def);
        }

        static public double GetDoubleParamByName(string paramName, string paramlist, double def)
        {
            string s = GetParamByName(paramName, paramlist);

            return StrToDoubleDef(s, def);
        }

        static public string GetLang(string lang)
        {
            if (string.IsNullOrEmpty(lang)) return "ru";
            return lang.ToLower().StartsWith("en") ? "en" : "ru";
        }

        static public string GetLangFromParamlist(string paramlist)
        {
            return GetLang(CommonUtils.GetParamByName("LANG", paramlist));
        }

        static public string ReplaceAll(string s, string from, string to)
        {
            string result = s.Replace(from, to);
            while (result != s)
            {
                s = result;
                result = s.Replace(from, to);
            }
            //s = result;
            return result;
        }
        static public void ReplaceAll(ref StringBuilder b, string from, string to)
        {
            int len;

            do
            {
                len = b.Length;
                b = b.Replace(from, to);
            }
            while (b.Length != len);
        }

        /*static public string ReplaceAll(string s, string from, string to, bool ignorecase)
        {
            if (ignorecase)
            {
                s = ReplaceAll(s, from, to);
                s = ReplaceAll(s, from.ToUpper(), to);
                s = ReplaceAll(s, from.ToLower(), to);
            }
            else
            {
                s = ReplaceAll(s, from, to);
            }
            return s;
        }*/


        static public bool Contains(this string s, string value, StringComparison comparison)
        {
            return s.IndexOf(value, comparison) >= 0;
        }
        static public int IndexOfAny(this string src, IEnumerable<string> list, StringComparison comparison)
        {
            throw new NotImplementedException("IndexOfAny - пока нигде не использовано");
            //для скорости можно при обнаружени в нулевой позиции сразу возвращать результат а не искать по всем строкам
            return list
                //.Where(s=> !s.IsNullOrEmpty()) //хотя эту проверку ненадо пусть выскочит стандартный эксепшин
                .Select(s => s.IndexOf(s, comparison))
                .Where(p => p >= 0)
                .DefaultIfEmpty(-1)
                .Min();
        }
        static public string SubstringBetween(this string s, int start, int end)
        {
            if (s == null) throw new ArgumentNullException();
            //тут правильно start < -1 но многое уже рассчитывает на -1 как отстутсвие слова. всетаки бкрем между слов которые должны существовать
            if (start < 0 || end < 0) return "";
            int len = end - start - 1;
            if (len <= 0) return "";
            return s.Substring(start + 1, len);
        }
        static public string SubstringBefore(this string s, int p)
        {
            if (s == null) throw new ArgumentNullException();
            if (p < 0) return "";
            return s.Substring(0, p);
        }
        static public string SubstringAfter(this string s, int p)
        {
            if (s == null) throw new ArgumentNullException();
            if (p < 0) return "";
            return s.Substring(p + 1);
        }
        static public string Replace(this string s, string oldValue, string newValue, StringComparison comparison, ReplaceCount count = ReplaceCount.All)
        {
            return Replace(s, oldValue, newValue, count, comparison);
        }
        static public string Replace(this string s, string oldValue, string newValue, ReplaceCount count, StringComparison comparison = StringComparison.Ordinal)
        {
            if (oldValue.IsNullOrEmpty()) throw new ArgumentException();

            if (count == ReplaceCount.All && comparison == StringComparison.Ordinal)
                return s.Replace(oldValue, newValue);

            int p;
            if (count == ReplaceCount.OnceLast)
                p = s.LastIndexOf(oldValue, comparison);
            else
                p = s.IndexOf(oldValue, comparison);

            if (p < 0)
                return s;

            //кеш для скорости чтоб по сто раз не лазить 
            int sLength = s.Length;
            int oldValueLength = oldValue.Length;
            int newValueLength = newValue.Length;

            //это нужно чтоб оптимизировать замену на ReplaceCount.Once если вхождение только одно. 
            int p2 = -1; //меньше нуля значит вхождение только одно и идем по короткому пути

            //флаг StringComparison.CurrentCulture означает что мы в любом случае идем по длинному пути. нужно для тестирования
            if (count == ReplaceCount.All && comparison != StringComparison.CurrentCulture)
            {
                p2 = s.IndexOf(oldValue, p + oldValueLength, comparison);
                if (p2 < 0) count = ReplaceCount.Once;
            }

            if (count == ReplaceCount.Once || count == ReplaceCount.OnceLast)
            {
                StringBuilder result = new StringBuilder(sLength + (newValueLength - oldValueLength));
                if (p > 0) result.Append(s, 0, p);

                //if (newValue.Length > 0) 
                result.Append(newValue);

                p += oldValueLength;
                //if (p < s.Length) 
                result.Append(s, p, sLength - p); //int.MaxValue не проканало

                return result.ToString();
                //return string.Concat(SubstringBefore(s, p), newValue, SubstringAfter(s, p));
            }

            if (count == ReplaceCount.All && comparison != StringComparison.Ordinal)
            {
                List<int> list = new List<int>(4);

                //все нормально работает как при наличии p2 так и приотсутствии
                if (p2 >= 0)
                {
                    list.Add(p);
                    p = p2;
                }

                do
                {
                    list.Add(p);
                    p = s.IndexOf(oldValue, p + oldValueLength, comparison);
                }
                while (p >= 0);


                StringBuilder result = new StringBuilder(sLength + list.Count * (newValueLength - oldValueLength));

                p = 0;
                foreach (int n in list)
                {
                    result.Append(s, p, n - p);

                    //if (newValue.Length > 0) 
                    result.Append(newValue);

                    p = n + oldValueLength;
                }
                //if (p < s.Length) 
                result.Append(s, p, sLength - p); //int.MaxValue не проканало

                return result.ToString();
            }

            throw new NotImplementedException("StringReplace - непредусмотренное состояние аргументов");
        }



        static public void SaveToFile(char[] content, Encoding enc, string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create);
            BinaryWriter bw = new BinaryWriter(fs, enc);
            bw.Write(content);
            bw.Close();
            fs.Close();
        }

        static public void SaveToFile(char[] content, string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create);
            Encoding enc = Encoding.GetEncoding("windows-1251");
            BinaryWriter bw = new BinaryWriter(fs, enc);
            bw.Write(content);
            bw.Close();
            fs.Close();
        }
        static public void SaveToFile(byte[] content, string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create);
            BinaryWriter bw = new BinaryWriter(fs);
            bw.Write(content);
            bw.Close();
            fs.Close();
        }

        public static string LoadFromFile(string filename)
        {
            // это пора заменять на File.ReadAllText(

            string result = "";
            Encoding enc = Encoding.GetEncoding("windows-1251");

            using (StreamReader streamReader = new StreamReader(filename, enc))
            {
                result = streamReader.ReadToEnd();
                streamReader.Close();
            }
            return result;
        }

        public static void SaveLogs(string message, string filename)
        {
            using (StreamWriter streamWriter = new StreamWriter(filename, true, Encoding.GetEncoding("windows-1251")))
            {
                streamWriter.WriteLine("[{0}:{1}] {2}", DateTime.Now.ToLongTimeString(), DateTime.Now.Millisecond.ToString(), message);
                streamWriter.Close();
            }
            return;
        }

        static public void SaveRequest(
                            string str,
                            //char[] bimages,
                            string ps,
                            string cookies,
                            string head,
                            int UploadsCount,
                            string SaveRequestPath)
        {

            Random r = new Random();
            string filename = (r.Next()).ToString();

            if (!(CommonUtils.isNull(SaveRequestPath).Length > 0))
            {
                throw new Exception("не корректный SaveRequestPath");
            }

            SaveToFile(str.ToCharArray(), SaveRequestPath + filename + ".str");
            SaveToFile(ps.ToCharArray(), SaveRequestPath + filename + ".ps");
            if (head.Length > 0) SaveToFile(head.ToCharArray(), SaveRequestPath + filename + ".head");
            //if (UploadsCount > 0) SaveToFile(bimages, SaveRequestPath + Filename + ".files");

            return;
        }

        public static int ExecSQLNonQuery(SqlConnection workConnection, string sqlText)
        {
            //SqlConnection workConnection = null;
            //if (workConnection == null) workConnection = new SqlConnection(adoCS);

            SqlCommand workCommand = new SqlCommand(sqlText, workConnection);
            if (workConnection.State == ConnectionState.Closed) workConnection.Open();
            return workCommand.ExecuteNonQuery();
        }

        public static SqlDataReader ExecSQL(SqlConnection workConnection, string sqlText)
        {
            //SqlConnection workConnection = null;
            //if (workConnection == null) workConnection = new SqlConnection(adoCS);

            SqlCommand workCommand = new SqlCommand(sqlText, workConnection);
            workConnection.Open();
            //SqlDataReader workReader =
            return workCommand.ExecuteReader(CommandBehavior.CloseConnection /* CommandBehavior.SingleResult */);

        }
        public static SqlDataReader ExecSQL(SqlCommand workCommand)
        {
            workCommand.Connection.Open();
            return workCommand.ExecuteReader(CommandBehavior.CloseConnection /* CommandBehavior.SingleResult */);
        }
        public static SqlDataReader ExecSingleSQL(SqlConnection workConnection, string sqlText)
        {
            //SqlConnection workConnection = null;
            //if (workConnection == null) workConnection = new SqlConnection(adoCS);
            //if (сonnection.State != ConnectionState.Open) сonnection.Open();

            SqlCommand workCommand = new SqlCommand(sqlText, workConnection);
            workConnection.Open();
            //SqlDataReader workReader = 
            return workCommand.ExecuteReader(CommandBehavior.CloseConnection & CommandBehavior.SingleResult & CommandBehavior.SingleRow);
        }
        public static SqlDataReader ExecSingleSQL(SqlCommand workCommand)
        {
            workCommand.Connection.Open();
            return workCommand.ExecuteReader(CommandBehavior.CloseConnection & CommandBehavior.SingleResult & CommandBehavior.SingleRow);
        }

        public static WebProxy HTTPProxy = null;

        public static string HttpGet(string url, int? timeout = 0)
        {
            HttpWebRequest httpRequest = (HttpWebRequest)WebRequest.Create(url);
            if (CommonUtils.HTTPProxy != null) httpRequest.Proxy = CommonUtils.HTTPProxy;

            httpRequest.Method = "GET";
            // здесь бы надо делать случайный выбор между несколькими агентами
            httpRequest.UserAgent = "Mozilla/4.0+(compatible;+MSIE+6.0;+Windows+NT+5.1;+SV1;+.NET+CLR+1.1.4322)";
            //httpRequest._contentType="application/x-www-form-urlencoded";
            httpRequest.Timeout = 20000;
            if (timeout > 0) httpRequest.Timeout = timeout.Value;


            //для https иногда выпадает ошибка //HttpWebRequest Не удалось создать защищенный канал SSL/TLS
            //
            //ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls | SecurityProtocolType.Tls11 | SecurityProtocolType.Tls12 | SecurityProtocolType.Ssl3;


            using (WebResponse httpResponse = httpRequest.GetResponse())
            {
                string encoding = "windows-1251";
                Encoding enc = Encoding.GetEncoding(encoding);
                string headers = httpResponse.Headers.ToString().ToLower();
                if (headers.IndexOf("charset=utf-8") > -1) enc = Encoding.GetEncoding("utf-8");
                else if (headers.IndexOf("charset=utf-16") > -1) enc = Encoding.GetEncoding("utf-16");

                using (StreamReader streamReader = new StreamReader(httpResponse.GetResponseStream(), enc))
                    return streamReader.ReadToEnd();
            }
        }

        public static string HttpPost(string dataurl)
        {
            // для замены ампа, при вызове не забывать Replace("&", "%26"); проверено, работает
            string url;
            string data;
            if (dataurl.IndexOf("?") > -1)
            {
                url = dataurl.Substring(0, dataurl.IndexOf("?"));
                data = dataurl.Substring(url.Length + 1, dataurl.Length - url.Length - 1);
            }
            else
            {
                url = dataurl;
                data = "";
            }

            HttpWebRequest httpRequest = (HttpWebRequest)WebRequest.Create(url);

            if (CommonUtils.HTTPProxy != null) httpRequest.Proxy = CommonUtils.HTTPProxy;

            httpRequest.Method = "POST";
            httpRequest.ContentType = "application/x-www-form-urlencoded";
            httpRequest.ContentLength = data.Length;

            // здесь бы надо делать случайный выбор между несколькими агентами
            httpRequest.UserAgent = "Mozilla/4.0+(compatible;+MSIE+6.0;+Windows+NT+5.1;+SV1;+.NET+CLR+1.1.4322)";
            //httpRequest._contentType="application/x-www-form-urlencoded";

#if DEBUG
            httpRequest.Timeout = 10000 * 100;
#else
			httpRequest.Timeout=10000;
#endif

            //ASCIIEncoding encoding = new ASCIIEncoding();
            byte[] databytes = Encoding.GetEncoding("windows-1251").GetBytes(data);
            Stream dataStream = httpRequest.GetRequestStream();
            dataStream.Write(databytes, 0, databytes.Length);
            dataStream.Close();

            WebResponse httpResponse = httpRequest.GetResponse();

            Encoding encoding = Encoding.GetEncoding("windows-1251");
            string headers = httpResponse.Headers.ToString().ToLower();
            if (headers.IndexOf("charset=utf-8") > -1) encoding = Encoding.GetEncoding("utf-8");
            else if (headers.IndexOf("charset=utf-16") > -1) encoding = Encoding.GetEncoding("utf-16");

            //StreamReader streamReader = new StreamReader(httpResponse.GetResponseStream(), Encoding.GetEncoding("utf-8"));
            StreamReader streamReader = new StreamReader(httpResponse.GetResponseStream(), encoding);
            //StreamReader streamReader = new StreamReader(httpResponse.GetResponseStream());

            string result = streamReader.ReadToEnd();
            streamReader.Close();
            httpResponse.Close();
            return result;
        }

        public static string UploadValues(string url, NameValueCollection data)
        {
            WebClient webClient = new WebClient();

            if (CommonUtils.HTTPProxy != null) webClient.Proxy = CommonUtils.HTTPProxy;

            //webClient.Encoding = Encoding.GetEncoding("utf-8");
            byte[] responseArray = webClient.UploadValues(url, "POST", data);
            string result = Encoding.GetEncoding("windows-1251").GetString(responseArray);
            return result;
        }

        public static void DownloadFile(string url, string fileName)
        {
            //if (CommonUtils.HTTPProxy != null) webClient.Proxy = CommonUtils.HTTPProxy;
            using (var client = new WebClient())
            {
                client.DownloadFile(url, fileName);
            }
        }

        static public NameValueCollection GetMailParams(NameValueCollection nameValueParamlist)
        {
            //NameValueCollection nameValueParamlist;

            //if (Request.HttpMethod == "GET")
            //    nameValueParamlist = Request.QueryString;
            //else
            //    nameValueParamlist = Request.Form;

            NameValueCollection result = new NameValueCollection(nameValueParamlist.Count);

            for (int i = 0; i < nameValueParamlist.Count; i++)
            {
                string name = CommonUtils.isNull(nameValueParamlist.GetKey(i)).ToUpper().Trim();
                if ((name != "") && (name != "IID") && (name != "SECTIONID") && (name != "AGENT") && (name != "PATHINFO") && (name != "STID") && (name != "HOTLOG") && (name != "FOR") && (name != "PARSEMODE") && (name != "REDIRECTTO") && (name != "ACTION") && (name != "zxcxczv"))
                    result.Add(name, CommonUtils.isNull(nameValueParamlist[i]).Trim());
            }
            return result;
        }


        //static public NameValueCollection GetMailParams(HttpRequest Request)
        //{
        //    NameValueCollection nameValueParamlist;

        //    if (Request.HttpMethod == "GET")
        //        nameValueParamlist = Request.QueryString;
        //    else
        //        nameValueParamlist = Request.Form;

        //    NameValueCollection result = new NameValueCollection(nameValueParamlist.Count);

        //    for (int i = 0; i < nameValueParamlist.Count; i++)
        //    {
        //        string name = CommonUtils.isNull(nameValueParamlist.GetKey(i)).ToUpper().Trim();
        //        if ((name != "")&&(name != "IID")&&(name != "SECTIONID")&&(name != "AGENT")&&(name != "PATHINFO")&&(name != "STID")&&(name != "HOTLOG")&&(name != "FOR")&&(name != "PARSEMODE")&&(name != "REDIRECTTO")&&(name != "ACTION")&&(name != "zxcxczv"))
        //            result.Add(name, CommonUtils.isNull(nameValueParamlist[i]).Trim());
        //    }
        //    return result;
        //}


        static public string MailParamsToMailText(NameValueCollection paramlist)
        {
            StringBuilder result = new StringBuilder(paramlist.Count * 16); //Request.ContentLength

            if (paramlist.Count < 1) result.Append("Пустое сообщение\n\n");

            for (int i = 0; i < paramlist.Count; i++)
            {
                string value = paramlist[i];
                string name = paramlist.GetKey(i);
                if (name.IndexOf('.') > -1) continue;

                // здесь бы name заменять на значение name+".title" если оно есть

                result.Append("[");
                result.Append(name);
                result.Append("]: \n");
                //result.Append('-', nameValueParamlist.GetKey(i).Length);
                //result.Append("\node");
                if (paramlist[i] != "")
                    result.Append(value);
                else
                    result.Append("(нет данных)");
                result.Append("\n\n");
            }

            return result.ToString();
        }

        static public string MailParamsToNameValueXML(NameValueCollection paramlist)
        {
            return MailParamsToNameValueXML(paramlist, false); // по умолчанию дополнительные параметры не показываем
        }
        static public string MailParamsToNameValueXML(NameValueCollection paramlist, bool showHelpParams)
        {
            StringWriter s = new StringWriter();
            XmlTextWriter writer = new XmlTextWriter(s);

            s.WriteLine(@"<?xml version='1.0' standalone='yes'?>");
            writer.Formatting = Formatting.Indented;
            writer.WriteStartElement("ParamDataSet");

            for (int i = 0; i < paramlist.Count; i++)
                if (paramlist[i].Trim() != "")
                {
                    string value = paramlist[i];
                    string name = CommonUtils.isNull(paramlist.GetKey(i), "").ToUpper();
                    if (",IID,STID,SECTIONID,DOCTYPE,REDIRECTMESSAGE,REDIRECTTO,AGENT,".IndexOf("," + name + ",") != -1) continue;

                    if (name.IndexOf('.') == -1)
                        writer.WriteStartElement("ParamTable");
                    else
                    {
                        if (showHelpParams == false) continue;
                        writer.WriteStartElement("HelpTable");
                    }

                    writer.WriteStartElement("Name");
                    writer.WriteString(name);
                    writer.WriteEndElement();

                    writer.WriteStartElement("Value");
                    writer.WriteString(value);
                    writer.WriteEndElement();

                    writer.WriteEndElement();
                }

            writer.WriteEndElement();

            return s.ToString();
        }

        static public void FillMails(System.Net.Mail.MailAddressCollection mailCollection, string mailString)
        {
            if (mailString.IndexOf(' ') > -1) mailString = mailString.Replace(" ", "");
            string[] mails = mailString.Split(new char[] { ',', ';' });
            foreach (string s in mails) if (s.Length > 0) mailCollection.Add(s);
        }

        static public bool IsBodyHtml(string body)
        {
            if (string.IsNullOrEmpty(body)) return false;

            int p1 = body.IndexOf("<");
            int p2 = body.IndexOf("<?xml");

            if (p1 == -1) return false; // тегов вообще нет
            if (p1 == p2) return false; // значит это xml

            string s = body.Substring(0, p1);
            s = s.Replace(" ", "").Replace("\t", "").Replace("\r", "").Replace("\n", "");
            if (s == string.Empty) return true;

            return false;
        }

        static public bool IsBodyXml(string body)
        {
            if (string.IsNullOrEmpty(body)) return false;

            int p1 = body.IndexOf("<");
            int p2 = body.IndexOf("<?xml");
            //int p3 = body.LastIndexOf(">");

            if (p1 == -1) return false; // тегов вообще нет
            if (p1 == p2) return true; // значит это xml

            return false;
        }




    }
}

