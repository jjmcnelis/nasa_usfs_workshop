import requests
import getpass
from IPython.display import clear_output, HTML


class EarthdataLogin(requests.Session):
    """
    Prompt user for Earthdata credentials repeatedly until auth success. Test
    by attempting to download 44kb Daymet granule. Source:
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"             # urs login url
    
    ERROR = "Login failed ({0}). Retry or register." # failure message
    
    TEST = ("https://daac.ornl.gov/daacdata/daymet/" # ORNL DAAC Daymet
            "Daymet_V3_Annual_Climatology/data/"     # granule to touch to
            "daymet_v3_prcp_annttl_2017_pr.tif")     # test authentication; 
    
    REGISTER = HTML(                                 # registration prompt
        "<p style='font-weight:bold'><a href=https://urs.earthdata.nasa.gov"
        "/users/new>Click here to register an Earthdata account.</a></p>")
                              
    
    def __init__(self):
        fails = 0
        
        while True:
            display(self.REGISTER) # register prompt
            
            username = input("Username: ")            # prompt for username
            password = getpass.getpass("Password: ")  # secure prompt for pw
            
            super().__init__()                        # init requests session
            self.auth = (username, password)          # add user,pw
            
            try:                                     
                response = self.get(self.TEST)        # try to grab TEST
                response.raise_for_status()           # raise for status>400
                clear_output()                        # clear output
                display("Login successful. Download with: session.get(url)")
                break

            except requests.exceptions.HTTPError as e:
                clear_output()                        # clear cell output
                fails += 1                            # +1 fail counter
                display(self.ERROR.format(str(fails)))# print failure msg

    
    def rebuild_auth(self, prepared_request, response):
        """
        Overrides from the library to keep headers when redirected to or 
        from the NASA auth host.
        """
        
        headers = prepared_request.headers
        url = prepared_request.url
 
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            
            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
                   
        #self.auth = None                    # purge username/password inputs
        return                              # return requests.Session object


### RUN ON IMPORT
session = EarthdataLogin()