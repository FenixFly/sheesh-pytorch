using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using Newtonsoft.Json;


namespace CardioSpike
{
    public partial class BaseHandler 
    {

        //public class rhythmogram
        //{

        //    public string id { get; set; } = "1";
        //    public DateTime timeString { get; set; }
        //    public int risk { get; set; } 
    
        //}




        protected virtual void LoginHandler()
        {
            var login = param("login");
            var password = param("password");

            patient patient = null;

            if (login == "testuser" && password == "111")
                patient = new patient() { id = "111", name = "Горбунков Семен Семеныч" , risk=50};

            if (login == "testdoctor" && password == "222")
                patient = new patient() { id = "222", name = "Профессор Мориарти", doctor = true };

            if (patient == null) throw new AccessViolationException("не верное имя пользователя или пароль");

            writejson(patient).End();

            //string jsontext = JsonConvert.SerializeObject(user, Formatting.Indented);
            //write(jsontext);


            //writeline(nameof(LoginHandler));
            //this.WriteParams();
            //writeline("ok");
            this.End();
        }
    }
}
