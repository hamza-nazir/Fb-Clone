import { useEffect,useState } from "react";
import api from "./Axios";
import { useNavigate } from 'react-router-dom';
export default function useAuth() {
    const [user,setUser]=useState({})
    const navigate=useNavigate()
  useEffect(() => {
    api.post("/auth")
      .then((res) => {
        if(res.data.success===false){
            navigate('/login')
        }else{
            setUser(res.data.user) 
        }
      })
      .catch((err) => {
        console.error(err.message);
      });
  }, []); 
  return user;
}
