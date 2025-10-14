import React from 'react'

const Friends = ({allFriends}) => {
    console.log(allFriends);
    return (
        <div className='row pt-2 mt-2' style={{backgroundColor:'#f2f5f9ff' ,boxShadow:'1px 1px 5px grey'}}>
          <div className='mx-auto col-lg-8 col-sm-8 mt-2 ' style={{height:'700px',backgroundColor:'white',boxShadow:'1px 1px 5px grey',borderRadius:'10px'}}>

            <h3 className='pt-3 ps-2'>Friends</h3>
            <hr />

            <div className='row p-2'>
                {allFriends?.map((user,index) => (
                    <div key={index} className="col-5 p-2 ms-5 mb-3 border rounded-3">
                        <div className='row'>
                          <div className="col-3 p-2">
                              {user?.profileImg!=null?(
                       <img className='ms-2 rounded' src={user?.profileImg} width={100} />

                            ):(
                       <img className='ms-2 rounded' src='/profile.png' width={100} />
                        
                            )
                        }
                      </div>
                      <div className="col" style={{display:'flex',justifyContent:'center',alignItems:'center'}}><b>{user?.name}</b></div>
                      <div className="col" style={{display:'flex',justifyContent:'center',alignItems:'center'}}>
                        <i className="fa-solid fa-ellipsis"></i>
                      </div>
                        </div>
                        
                    </div>
                ))}
            </div>

          </div>
           </div>
    )
}

export default Friends
