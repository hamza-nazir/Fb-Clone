import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import useAuth from '../Hook/useAuth';
import api from '../Hook/Axios'
import CloseIcon from "@mui/icons-material/Close";
import { useContext, useEffect,useState } from "react";
import { Context } from "../Hook/Context";

const RequestComp = () => {
     const {request,setRequest}=useContext(Context);
    const [pendingRequest,setPendingRequest]=useState([]);


    useEffect(()=>{
    api.post('/get-request')
    .then((res)=>{
        if(res.data.checkRequestDocument!=null){
       setPendingRequest(res.data.checkRequestDocument.requests);
         }else{
             console.log( res.data);
              setPendingRequest(null)
         }
    })
    },[])


function acceptRequest(userToAcceptRequest){

api.post('/accept-request',{userToAcceptRequest})
.then((res)=>{
  setPendingRequest(res.data.currentUserDoc.requests);
})
}

    return (
        <div>
             <Drawer
                  open={request}
                  anchor="right"
                  variant="persistent"
                  sx={{
                    ".MuiDrawer-paper": {
                     padding:'10px',
                     marginRight:'30px',
                     width: 370,
                     height: 400,
                     marginTop: "50px",
                     boxShadow: "1px 1px 5px #96989bff",
                     borderRadius: "10px",
                    }
                 }}>
            <h3>Pending Requests</h3>

         <IconButton onClick={()=>setRequest(false)}
          sx={{ position: "absolute", right: 8, top: 10 }}>
        <CloseIcon />
      </IconButton>

      <hr />

    

{
    pendingRequest!=null&&pendingRequest.length>0?(
          

            pendingRequest?.map((user,index) => (

             
            <div  key={index} className="row border p-2 mt-1" style={{backgroundColor:'#F2F4F7'}}>
                <div className="col-2">
                    <img src="/profile.png" style={{borderRadius:'50%',border:'1px solid black'}} width={40} />
                </div>
               <div className="col-5 mt-1"> <b>{user?.name}</b> </div>

               <div className="col-5">
                <div className="row pe-2">
                    <div className="col-6"><button onClick={()=>acceptRequest(user._id)} className="btn btn-sm btn-primary">Accept</button></div>
                    <div className="col-6"><button className="btn btn-sm btn-primary">Delete</button></div>

                </div>
               </div>
            </div>
          
          
        ))
    ):(

      <h5 className="text-muted">  No Pending Request </h5>  
    )
}

          
       
     
          </Drawer>
         
        </div>
    )
}

export default RequestComp
