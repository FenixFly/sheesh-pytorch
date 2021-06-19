using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web;

namespace CardioSpike
{
    static public class Utils
    {


        static public string NormalizeCsv(this string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return "";

            IEnumerable<string> list = text.Split('\n', '\r');
            StringBuilder b = new StringBuilder(text.Length);

            list = list.Select(s => s.Trim().Replace(';', ','))
                .Where(s => !s.IsNullOrWhiteSpace());

            foreach (var s in list) b.AppendLine(s);
            
            return b.ToString();
        }

        static public Uri Extract404(this Uri url)
        {
            var urloriginalstring = url.OriginalString.Replace("https://cardiospike.ip3.ru:443/error404.aspx?404;", "");
            if (urloriginalstring != url.OriginalString) url = new Uri(urloriginalstring);
            return url;

        }

        //static public bool EqualPath(this Uri url, string path)
        //{
        //    var abspath = url.AbsolutePath;
        //    return abspath.EqualPath(path)
        //}
        //static public bool EqualPath(this string url, string path)
        //{
        //    var abspath = url;
        //    if (abspath.EndsWith("/default.aspx")) abspath = abspath.Replace("/default.aspx", "/");

        //    return path == abspath;
        //}


        static public HttpResponse writeline(this HttpResponse response,  string s = null)
        {
            if (s != null) response.Write(s);
            response.Write("<br/>\n");
            return response;
        }
        static public HttpResponse write(this HttpResponse response, string s)
        {
            response.Write(s);
            return response;
        }


    }
}
