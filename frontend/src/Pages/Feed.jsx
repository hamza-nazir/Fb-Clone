import useAuth from '../Hook/useAuth';
import DrawerComp from '../Components/DrawerComp';
import PostPopup from '../Components/PostPopup';
import '../css/Feed.css';
import Posts from '../Components/Posts';
import { useEffect } from 'react';
import { useContext } from 'react';
import { Context } from '../Hook/Context';
import LogoutPop from '../Components/LogoutPop';
import RequestComp from '../Components/RequestComp';
const Feed = () => {

      const user = useAuth();
          const {setPostPopUp}=useContext(Context)

    return (
      <>
     
       <div className='mx-auto mt-3' style={{backgroundColor:'white',width:'75%',borderRadius:'10Px'}}>
         <div className='row p-2' >
          <div className='mb-2 pt-2' style={{display:'flex',justifyContent:'center',alignItems:'center'}}>
             <div className="col-2 ">
              {
                user?.profileImg?(
            <img style={{borderRadius:'50%',height:'40px'}} src={user?.profileImg} width={40} />

                ):(
            <img style={{borderRadius:'50%',height:'40px'}} src="profile.png" width={40} />

                )}
           </div>

           <div   className="col-10">
            
            <input readOnly onClick={()=>setPostPopUp(true)}  id='feed-input' placeholder={`Whats on your mind, ${user?.name} ?`} type="text" style={{width:'100%',padding:'6px',border:'none',borderRadius:'15px',backgroundColor:'#E5E7EA'}} />
           </div>
          </div>
      
           <hr className='mx-auto' style={{width:'95%'}} />
           <div className="row text-center pb-2">
            <div className="col"><i style={{color:'red'}} className="fa-solid fa-video me-2"></i><span className='text-muted' >Live video</span></div>
            <div className="col"><i style={{color:'#41B35D'}} className="fa-solid fa-image me-2"></i><span className='text-muted'>Photo</span></div>
            <div className="col"><i style={{color:'#EBB22D'}} className="fa-solid fa-face-smile me-2"></i><span className='text-muted'>Feeling</span></div>
           </div>
        </div>

       </div>

        <div className='mx-auto mt-3' style={{width:'75%'}}>

        <Posts/>

        </div>
      <DrawerComp/>

      <PostPopup/>
      <LogoutPop/>
      <RequestComp/>
      </>
    )
}

export default Feed
