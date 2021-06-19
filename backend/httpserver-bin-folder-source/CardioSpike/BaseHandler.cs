using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using Newtonsoft.Json;


namespace CardioSpike
{
    public partial class BaseHandler : System.Web.UI.Page
    {

        static public string appdatapath()
        {
            var workpath = @"c:\Work\Test\CardioSpike\App_Data\";
            var srvpath = @"C:\inetpub\SITE\test\cardiospike\App_Data\";

            if (Directory.Exists(srvpath)) return srvpath;

            return workpath;

            //string path = Server.MapPath("~/App_Data/");
            //string path = @"C:\inetpub\SITE\test\cardiospike\App_Data\";
            //return path;
        }
        static public string appdatapath(string subpath)
        {
            return Path.Combine(appdatapath(), subpath);
        }

        static public string filestoragepath()
        {
            var workpath = @"c:\Work\Test\CardioSpike\filestorage\";
            var srvpath = @"C:\inetpub\SITE\test\cardiospike\filestorage\";

            if (Directory.Exists(srvpath)) return srvpath;

            return workpath;
        }

        static public string filestoragepath(string subpath)
        {
            return Path.Combine(filestoragepath(), subpath);
        }

        public string param(string name)
        {
            var val = this.Request.Params[name];
            return val;
        }

        public HttpResponse writejson(object obj)
        {
            string jsontext = JsonConvert.SerializeObject(obj, Formatting.Indented);
            return this.write(jsontext);
        }

        public HttpResponse writeline(string s = null)
        {
            return this.Response.writeline(s);
        }
        public HttpResponse write(string s)
        {
            return this.Response.write(s);
        }
        public void End()
        {
            this.Response.End();
        }

        public void WriteUrl(Uri url = null)
        {
            if (url == null) url = url = this.Request.Url.Extract404();

            //https://docs.microsoft.com/ru-ru/dotnet/api/system.uri?view=netframework-4.8

            //Uri uri = new Uri("https://user:password@www.contoso.com:80/Home/Index.htm?q1=v1&q2=v2#FragmentName");
            // AbsolutePath: /Home/Index.htm
            // AbsoluteUri: https://user:password@www.contoso.com:80/Home/Index.htm?q1=v1&q2=v2#FragmentName
            // DnsSafeHost: www.contoso.com
            // Fragment: #FragmentName
            // Host: www.contoso.com
            // HostNameType: Dns
            // IdnHost: www.contoso.com
            // IsAbsoluteUri: True
            // IsDefaultPort: False
            // IsFile: False
            // IsLoopback: False
            // IsUnc: False
            // LocalPath: /Home/Index.htm
            // OriginalString: https://user:password@www.contoso.com:80/Home/Index.htm?q1=v1&q2=v2#FragmentName
            // PathAndQuery: /Home/Index.htm?q1=v1&q2=v2
            // Port: 80
            // Query: ?q1=v1&q2=v2
            // Scheme: https
            // Segments: /, Home/, Index.htm
            // UserEscaped: False
            // UserInfo: user:password


            writeline($"AbsolutePath: {url.AbsolutePath}");
            writeline($"AbsoluteUri: {url.AbsoluteUri}");
            writeline($"DnsSafeHost: {url.DnsSafeHost}");
            writeline($"Fragment: {url.Fragment}");
            writeline($"Host: {url.Host}");
            writeline($"HostNameType: {url.HostNameType}");
            writeline($"IdnHost: {url.IdnHost}");
            writeline($"IsAbsoluteUri: {url.IsAbsoluteUri}");
            writeline($"IsDefaultPort: {url.IsDefaultPort}");
            writeline($"IsFile: {url.IsFile}");
            writeline($"IsLoopback: {url.IsLoopback}");
            writeline($"IsUnc: {url.IsUnc}");
            writeline($"LocalPath: {url.LocalPath}");
            writeline($"OriginalString: {url.OriginalString}");
            writeline($"PathAndQuery: {url.PathAndQuery}");
            writeline($"Port: {url.Port}");
            writeline($"Query: {url.Query}");
            writeline($"Scheme: {url.Scheme}");
            writeline($"Segments: {string.Join(", ", url.Segments)}");
            writeline($"UserEscaped: {url.UserEscaped}");
            writeline($"UserInfo: {url.UserInfo}");
            writeline();
            if (this.Request.Url.ToString() != url.ToString())
                write("//перенеправлен").writeline(this.Request.Url.ToString());
        }

        public void WriteParams()
        {
            //Response.Write(this.GetType().FullName);
            //Response.Write("<br/>");
            writeline(this.Request.Url.Extract404().ToString());
            writeline();

            //https://cardiospike.ip3.ru/test.aspx
            foreach (var p in this.Request.Params.AllKeys)
            {
                if (string.IsNullOrWhiteSpace(p)) continue;
                var v = this.Request.Params[p];

                writeline($"{p}={v}");
            }

        }



        protected virtual void RouteHandler()
        {

            if (!this.Route())
            {
                //https://cardiospike.ip3.ru:443/error404.aspx?404;

                this.ContentType = "text/html";
                this.Response.ContentType = this.ContentType;
                this.Response.AppendHeader("Content-Type", this.ContentType);
                this.Response.Status = "404 Not Found";
                this.Response.StatusCode = 404;


                writeline("<h1>Error 404</h1>");
                writeline("<h2>страница или боработчик не найден</h2>");
                writeline("");
                writeline("");
                writeline("");

                this.WriteUrl();
                this.WriteParams();

                this.End();
            }
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            RouteHandler();
        }

    }
}
