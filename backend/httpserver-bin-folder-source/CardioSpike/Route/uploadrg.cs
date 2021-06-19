using CsvHelper;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using System.Web.Management;

namespace CardioSpike
{
    public partial class BaseHandler 
    {
        protected virtual void UploadRGHandler()
        {
            var id = param("id");
            var incsv = param("rgtext");

            HttpFileCollection filelist = this.Request.Files;
            if (filelist.Count > 1) throw new Exception("нельзя прикреплять несколько файлов");
            if (filelist.Count == 1 && filelist[0].ContentLength > 0)
            {
                //this.SaveLogs(Files[a].ContentType);
                HttpPostedFile file = filelist[0];

                if (!file.FileName.IsNullOrWhiteSpace())
                {
                    var ext = Path.GetExtension(file.FileName);
                    if (!ext.Equals(".csv", StringComparison.OrdinalIgnoreCase))
                        throw new Exception("надо загружать .csv файл");
                }
                
                using (StreamReader sr = new StreamReader(file.InputStream))
                {
                    var text = sr.ReadToEnd();
                    if (!text.IsNullOrWhiteSpace()) incsv = text;
                }
               
            }

            if (id.IsNullOrWhiteSpace()) throw new Exception("отсутствует id");
            if (incsv.IsNullOrWhiteSpace()) throw new Exception("отсутствует csv");
            incsv = incsv.NormalizeCsv();
            if (!incsv.StartsWith("time,x")) throw new Exception("не тот формат файла который нужен");


            var outcsv = PostToEvgenii(incsv);
            string guid = Guid.NewGuid().ToString();
            string rid = $"{id}@{guid}";            
            string csvpath = filestoragepath(Path.Combine(id, guid + ".csv"));
            File.WriteAllText(csvpath, outcsv);

            var rhythmogram = RhythmogramFromCsv(outcsv);
            rhythmogram.id = rid;
            var jsontext = Converter.tojson(rhythmogram);

            write(jsontext).End();

            //var jsontext = RhythmogramCsvToJson(id, outcsv);
            //write(jsontext).End();


            //= UploadRG(id, incsv);

            //writeline(outcsv);

            //writeline(nameof(UploadRGHandler));
            //this.WriteParams();
            //writeline("ok");
            this.End();
        }








        static public string PostToEvgenii(string data)
        {
            if (data.IsNullOrWhiteSpace()) throw new Exception("пустой текст постить не Евгению не надо");
            if (!data.StartsWith("time,x")) throw new Exception("не тот формат который нужен Евгению");

            var url = "http://95.79.25.36:45678/";
            string result = null;
            
            var httpRequest = (System.Net.HttpWebRequest)System.Net.WebRequest.Create(url);
            httpRequest.Method = "POST";

            httpRequest.ContentType = "text/plain";

            using (var streamWriter = new StreamWriter(httpRequest.GetRequestStream()))
            {
                streamWriter.Write(data);
            }

            var httpResponse = (System.Net.HttpWebResponse)httpRequest.GetResponse();
            using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                result = streamReader.ReadToEnd();

            result = result.NormalizeCsv();
            //Console.WriteLine(httpResponse.StatusCode);


            //Evgenii
            return result;

        }




        static public string UploadRG(string id, string incsv)
        {
            return incsv;
        }



    }



}
