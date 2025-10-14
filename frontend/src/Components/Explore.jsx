import '../css/Explore.css'
import useAuth from '../Hook/useAuth'
import { Link } from 'react-router-dom';


const Explore = () => {

  const currUser=useAuth();

    return (
        <div   style={{height:'90vh',display:'flex',flexDirection:'column',justifyContent:'space-between'}}>
            <Link style={{textDecoration:'none',color:'black'}} to={`/user/${currUser?._id}`} className='mt-5 explore-div p-2'>
            {
              currUser?.profileImg!=null?(
                <img className='me-3'  src={currUser?.profileImg} style={{borderRadius:'50%',height:'45px',width:'45px',objectFit:'cover'}} />

              ):(
                <img className='me-3'  src="/profile.png" style={{borderRadius:'50%',height:'40px',width:'40px',boxShadow:'1px 1px 5px #cacbceff'}} />

              )
            }
               
                <span ><b >{currUser?.name}</b></span>
                </Link> 

            <div className='explore-div p-2'>
                <img className='me-3'  src="https://cdn-1.webcatalog.io/catalog/meta-ai/meta-ai-icon-filled-256.webp?v=1714783130413" style={{borderRadius:'50%',height:'40px',width:'40px'}}  />
              <span> <b> Meta AI</b></span>
                </div>

          <div className='explore-div p-2'>
                <img className='me-3'  src='watch.jpg' style={{borderRadius:'50%',height:'38px',width:'38px'}}  />
              <span> <b>Reels</b></span>
                </div>

            <div className='explore-div p-2'>
                <img  className='me-3' src='friends.jpg' style={{borderRadius:'50%',height:'40px',width:'40px'}}  />
              <span> <b> Friends</b></span>
                </div>

          <div className='explore-div p-2'>
                <img className='me-3' src='market.jpg' style={{borderRadius:'50%',height:'38px',width:'38px'}}  />
              <span> <b>Market</b></span>
                </div>

                  <div className='explore-div p-2'>
                <img className='me-3'  src="group.jpg" style={{borderRadius:'50%',height:'40px',width:'40px'}}  />
              <span> <b> Groups</b></span>
                </div>

          <div className='explore-div p-2'>
                <img className='me-3'  src='heart.jpg' style={{borderRadius:'50%',height:'38px',width:'38px'}}  />
              <span> <b>Liked</b></span>
                </div>

                  <div className='explore-div p-2'>
                <img className='me-3'  src="live.jpg" style={{borderRadius:'50%',height:'40px',width:'40px'}}  />
              <span> <b>Live</b></span>
                </div>

          <div className='explore-div p-2'>
                <img className='me-3'  src='game.jpg' style={{borderRadius:'50%',height:'38px',width:'38px'}}  />
              <span> <b>Games</b></span>
                </div>
               

         
   

        </div>
    )
}

export default Explore
