using CsvHelper;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;

namespace CardioSpike
{
    public partial class BaseHandler 
    {

        //public class patientlist : List<patient> { }

        protected virtual void PatientsHandler()
        {
            var list = GetPatientList();

            writejson(list).End();


            writeline(nameof(PatientsHandler));
            writeline(filestoragepath());
            this.End();
        }

        static public List<patient> GetPatientList()
        {
            //var dirlist = Directory.EnumerateDirectories(filestoragepath().Replace(@"App_Data\","")).Select(s=>new DirectoryInfo(s)).ToList();
            var dirlist = Directory.EnumerateDirectories(filestoragepath())
                .Select(s => new DirectoryInfo(s))
                .OrderByDescending(d => d.LastWriteTime)
                .ToList();

            var patientlist = new List<patient>();
            foreach(var dir in dirlist)
            {
                var p = new patient();
                p.id = dir.Name;
                p.name = dir.Name;
                if (Converter.ToInt(p.name) > 0) p.name = "Пациент №" + dir.Name;

                p.lastRgTimeString = dir.LastWriteTime.ToString("g");
                patientlist.Add(p);
            }
            return patientlist;
        }

        static public void testGetPatientList()
        {
            var list = GetPatientList().OrderByDescending(d => d.lastRgTimeString).ToList();
            if (list.Count < 150) throw new NotImplementedException();

            var s = Converter.tojson(list);
        }
    }
}
