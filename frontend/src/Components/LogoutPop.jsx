import Drawer from "@mui/material/Drawer";
import { useContext } from "react";
import { Context } from "../Hook/Context";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import api from '../Hook/Axios'
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Divider from "@mui/material/Divider";
import Box from "@mui/material/Box";
import LogoutIcon from "@mui/icons-material/Logout";
import SettingsIcon from "@mui/icons-material/Settings";
import Swal from "sweetalert2";
import PrivacyTipIcon from "@mui/icons-material/PrivacyTip";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import { useNavigate } from 'react-router-dom';

const LogoutPop = () => {
    const navigate=useNavigate();
  const { logout, setLogout } = useContext(Context);

  function logoutFun(){
    setLogout(false)
     Swal.fire({
    title: "Are you sure?",
    text: "You will be logged out of your account.",
    icon: "warning",
    showCancelButton: true,
    confirmButtonColor: "#0866FF",
    cancelButtonColor: "grey",
    confirmButtonText: "Yes, logout",
  })
  .then((res)=>{
    if(res.isConfirmed){
        api.post('/logout')
        .then((res)=>{
            if(res.data.success==true){
                navigate('/login')
            }
        })
    }
  })
  }
  return (
    <Drawer
      open={logout}
      anchor="right"
      variant="persistent"
      sx={{
        ".MuiDrawer-paper": {
          width: 319,
          height: 300,
          marginTop: "50px",
          boxShadow: "1px 1px 5px #96989bff",
          borderRadius: "8px 0 0 8px",
        },
      }}
    >
    
      <IconButton
        onClick={() => setLogout(false)}
        sx={{ position: "absolute", right: 8, top: 10 }}
      >
        <CloseIcon />
      </IconButton>


      <Box sx={{ mt: 6 }}>
        <List>
          <ListItemButton onClick={logoutFun}>
            <ListItemIcon>
              <LogoutIcon color="primary" />
            </ListItemIcon>
            <ListItemText primary="Logout" />
          </ListItemButton>

          <Divider />

          <ListItemButton>
            <ListItemIcon>
              <SettingsIcon color="action" />
            </ListItemIcon>
            <ListItemText primary="Settings & Privacy" />
          </ListItemButton>

          <ListItemButton>
            <ListItemIcon>
              <PrivacyTipIcon color="action" />
            </ListItemIcon>
            <ListItemText primary="Security & Privacy" />
          </ListItemButton>

          <ListItemButton>
            <ListItemIcon>
              <HelpOutlineIcon color="action" />
            </ListItemIcon>
            <ListItemText primary="Help & Support" />
          </ListItemButton>
        </List>
      </Box>
    </Drawer>
  );
};

export default LogoutPop;
