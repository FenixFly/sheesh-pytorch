using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;

namespace CardioSpike
{
    public partial class BaseHandler : System.Web.UI.Page
    {

        protected virtual bool Route()
        {
            var url = this.Request.Url.Extract404();
            var path = url.AbsolutePath;
            if (path.EndsWith("/default.aspx")) path = path.Replace("/default.aspx", "/");


            if (path == "/test/")
            {
                this.TestHandler();
            }
            else if (path == "/login/")
            {
                this.LoginHandler();
            }
            else if (path == "/uploadrg/")
            {
                this.UploadRGHandler();
            }
            else if (path == "/patients/")
            {
                this.PatientsHandler();
            }
            else if (path == "/patient/")
            {
                this.PatientHandler();
            }
            else if (path == "/rg/")
            {
                this.RGHandler();
            }

            else if (url.ToString().Contains("test34234634584"))
            {
                writeline("ok test34234634584");
                this.End();
            }
            return false;
        }






    }
}
