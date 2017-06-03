document.addEventListener('DOMContentLoaded', () => {
  const CellSize = 16

  const defaultWidth = Math.floor(window.innerWidth / CellSize)
  const defaultHeight = Math.floor(window.innerHeight / CellSize)

  const cells = []
  const grid = document.createElement('table')
  grid.classList.add('grid')
  document.body.appendChild(grid)

  let initialized = false
  let width, height

  const socket = io(window.location.toString(), window.socketioOptions)
  let state = undefined
  socket.on('connect', () => {
    if (state) {
      socket.emit('next', state)
    } else {
      socket.emit('start', {width: defaultWidth, height: defaultHeight})
    }
  })
  socket.on('generation', newState => {
    height = newState.length
    width = newState.length > 0 ? newState[0].length : 0

    if (!initialized) {
      for (let y = 0; y < height; y++) {
        const row = []
        const gridRow = document.createElement('tr')
        for (let x = 0; x < width; x++) {
          const cell = document.createElement('td')
          cell.classList.add('cell', 'dead')
          row.push(cell)
          gridRow.appendChild(cell)
        }
        cells.push(row)
        grid.appendChild(gridRow)
      }
      initialized = true
    }

    state = newState
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (state[y][x]) {
          cells[y][x].classList.remove('dead')
          cells[y][x].classList.add('alive')
        } else {
          cells[y][x].classList.remove('alive')
          cells[y][x].classList.add('dead')
        }
      }
    }
    setTimeout(() => socket.emit('next', state), 100)
  })
})
