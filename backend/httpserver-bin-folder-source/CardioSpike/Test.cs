using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CardioSpike
{
    public class Test
    {
        static public void Start()
        {
            //BaseHandler.testPostToEvgenii(); //проверил работает
            BaseHandler.testEvgeniiCsvToJson();
            BaseHandler.testGetPatientList();
        }


    }
}
